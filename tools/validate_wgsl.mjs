// Compiles every shader under shader/webgpu with the WGSL compiler of a
// headless Chrome and reports errors and warnings.
//
// Usage:
//   node tools/validate_wgsl.mjs

import { spawn } from 'child_process';
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
} from 'fs';
import { createServer } from 'http';
import { tmpdir } from 'os';
import { basename, dirname, join } from 'path';
import { fileURLToPath } from 'url';
import puppeteer from 'puppeteer-core';

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));

const CHROME_ARGS = [
  '--headless=new',
  '--no-sandbox',
  '--enable-unsafe-webgpu',
  '--use-angle=swiftshader',
  '--enable-features=Vulkan,VulkanFromANGLE',
  '--disable-vulkan-surface',
];

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
];

function findChrome() {
  for (const candidate of CHROME_CANDIDATES) {
    if (candidate && existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(
    'Chrome binary is not found. Set the CHROME_PATH environment variable.'
  );
}

function collectShaders(dir) {
  const shaders = [];
  for (const entry of readdirSync(dir).sort()) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) {
      shaders.push(...collectShaders(p));
    } else if (entry.endsWith('.wgsl')) {
      shaders.push({
        name: basename(entry, '.wgsl'),
        path: p,
        source: readFileSync(p, { encoding: 'utf-8' }),
      });
    }
  }
  return shaders;
}

async function startChrome(executablePath, userDataDir) {
  const chrome = spawn(
    executablePath,
    [
      ...CHROME_ARGS,
      '--remote-debugging-port=0',
      `--user-data-dir=${userDataDir}`,
      'about:blank',
    ],
    { stdio: ['ignore', 'pipe', 'pipe'] }
  );
  const endpoint = await new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('Chrome did not report a DevTools endpoint')),
      30000
    );
    chrome.stderr.on('data', (chunk) => {
      const m = /ws:\/\/[^\s]+/.exec(chunk.toString());
      if (m) {
        clearTimeout(timer);
        resolve(m[0]);
      }
    });
  });
  return { chrome, endpoint };
}

const shaders = collectShaders(join(packageRoot, 'shader/webgpu'));
if (shaders.length === 0) {
  throw new Error('No .wgsl file was found');
}
console.log(`Validating ${shaders.length} shaders`);

// WebGPU is not available on an opaque origin such as about:blank, so a blank
// page is served over http.
const server = createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end('<!DOCTYPE html><html><body></body></html>');
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const pageUrl = `http://127.0.0.1:${server.address().port}/`;

const userDataDir = mkdtempSync(join(tmpdir(), 'distmljs-wgsl-'));
const { chrome, endpoint } = await startChrome(findChrome(), userDataDir);
const browser = await puppeteer.connect({ browserWSEndpoint: endpoint });
try {
  const page = await browser.newPage();
  await page.goto(pageUrl);

  // The GPU process may still be starting up, so requesting the adapter and the
  // device is retried as a pair.
  const ready = await page.evaluate(async () => {
    for (let i = 0; i < 40; i++) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          window.__device = await adapter.requestDevice();
          return true;
        }
      } catch {
        // retry
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return false;
  });
  if (!ready) {
    throw new Error('Could not obtain a GPUDevice');
  }

  let errorCount = 0;
  let warningCount = 0;
  for (const shader of shaders) {
    const messages = await page.evaluate(async (source) => {
      const device = window.__device;
      device.pushErrorScope('validation');
      const module = device.createShaderModule({ code: source });
      const info = await module.getCompilationInfo();
      await device.popErrorScope();
      return info.messages.map((m) => ({
        type: m.type,
        lineNum: m.lineNum,
        message: m.message,
      }));
    }, shader.source);
    for (const message of messages) {
      if (message.type === 'error') {
        errorCount++;
      } else if (message.type === 'warning') {
        warningCount++;
      }
      console.log(
        `${message.type}: ${shader.path}:${message.lineNum}: ${message.message}`
      );
    }
  }
  console.log('');
  console.log(
    `${shaders.length} shaders, ${errorCount} errors, ${warningCount} warnings`
  );
  process.exitCode = errorCount > 0 ? 1 : 0;
} finally {
  await browser.disconnect();
  chrome.kill();
  await new Promise((resolve) => chrome.once('exit', resolve));
  rmSync(userDataDir, { recursive: true, force: true });
  server.close();
}
