// Runs the mocha test suite in a headless Chrome and reports the result.
//
// Usage:
//   node tools/run_browser_test.mjs [--target=webgl,webgpu,heavy] [--headful] [--port=8123]
//
// The Chrome binary is looked up in the CHROME_PATH environment variable,
// falling back to the well-known locations of Google Chrome / Chromium.

import { spawn } from 'child_process';
import { existsSync, mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import puppeteer from 'puppeteer-core';

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));

// SwiftShader (software implementation of Vulkan) is used so that the test runs
// on a machine without GPU. --no-sandbox is required to run inside a container.
const CHROME_ARGS = [
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

function parseArgs(argv) {
  const options = { target: '', headful: false, port: 8123, timeout: 1800 };
  for (const arg of argv) {
    const m = /^--([^=]+)(?:=(.*))?$/.exec(arg);
    if (!m) {
      throw new Error(`Unknown argument: ${arg}`);
    }
    const [, key, value] = m;
    switch (key) {
      case 'target':
        options.target = value || '';
        break;
      case 'headful':
        options.headful = true;
        break;
      case 'port':
        options.port = Number(value);
        break;
      case 'timeout':
        options.timeout = Number(value);
        break;
      default:
        throw new Error(`Unknown option: --${key}`);
    }
  }
  return options;
}

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

function waitForExit(child) {
  return new Promise((resolve) => child.once('exit', resolve));
}

async function startHttpServer(port) {
  const server = spawn(
    process.platform === 'win32' ? 'npx.cmd' : 'npx',
    ['http-server', '-c-1', '-p', String(port), '--silent'],
    { cwd: packageRoot, stdio: 'ignore' }
  );
  // Wait until the port accepts a request.
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/test/`);
      if (res.ok) {
        return server;
      }
    } catch {
      // not listening yet
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  server.kill();
  throw new Error(`http-server did not start on port ${port}`);
}

async function startChrome(executablePath, headful, userDataDir) {
  const args = [
    ...(headful ? [] : ['--headless=new']),
    ...CHROME_ARGS,
    '--remote-debugging-port=0',
    `--user-data-dir=${userDataDir}`,
    'about:blank',
  ];
  const chrome = spawn(executablePath, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
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
    chrome.once('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`Chrome exited with code ${code}`));
    });
  });
  return { chrome, endpoint };
}

async function collectResult(page) {
  return page.evaluate(() => {
    const text = (selector) => {
      const el = document.querySelector(selector);
      return el ? el.textContent : '0';
    };
    // The mocha HTML reporter nests each test under li.suite elements, so the
    // enclosing suite titles are collected to make the test identifiable.
    const suitePath = (el) => {
      const titles = [];
      for (let p = el.parentElement; p; p = p.parentElement) {
        if (p.classList && p.classList.contains('suite')) {
          const h1 = p.querySelector('h1');
          if (h1) {
            titles.unshift(h1.textContent.trim());
          }
        }
      }
      return titles;
    };
    const failures = Array.from(
      document.querySelectorAll('#mocha .test.fail')
    ).map((el) => {
      const title = el.querySelector('h2');
      const error = el.querySelector('.error');
      return {
        title: [
          ...suitePath(el),
          title ? title.childNodes[0].textContent.trim() : '(unknown)',
        ].join(' > '),
        error: error ? error.textContent.split('\n')[0] : '',
      };
    });
    const initErrors = Array.from(
      document.querySelectorAll('#error p')
    ).map((el) => el.textContent);
    return {
      passes: Number(text('#mocha-stats .passes em')),
      reportedFailures: Number(text('#mocha-stats .failures em')),
      total: document.querySelectorAll('#mocha .test').length,
      failures,
      initErrors,
    };
  });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const executablePath = findChrome();
  const userDataDir = mkdtempSync(join(tmpdir(), 'distmljs-test-'));

  let server = null;
  let chrome = null;
  let browser = null;
  try {
    server = await startHttpServer(options.port);
    const started = await startChrome(
      executablePath,
      options.headful,
      userDataDir
    );
    chrome = started.chrome;
    browser = await puppeteer.connect({ browserWSEndpoint: started.endpoint });

    const page = await browser.newPage();
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));
    page.on('console', (msg) => {
      const url = msg.location() ? msg.location().url || '' : '';
      if (msg.type() === 'error' && !url.endsWith('/favicon.ico')) {
        pageErrors.push(msg.text());
      }
    });

    // target must always be given explicitly; the page defaults to webgl when
    // the query parameter is absent.
    const url = `http://127.0.0.1:${options.port}/test/?target=${options.target}`;
    console.log(`Running ${url}`);
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__mochaDone === true, {
      timeout: options.timeout * 1000,
      polling: 1000,
    });

    const result = await collectResult(page);
    console.log('');
    for (const initError of result.initErrors) {
      console.log(`INIT ERROR: ${initError}`);
    }
    for (const failure of result.failures) {
      console.log(`FAIL: ${failure.title} -- ${failure.error}`);
    }
    console.log('');
    console.log(
      `target=${options.target || '(cpu only)'} total=${result.total} ` +
        `passes=${result.passes} failures=${result.reportedFailures}`
    );
    if (pageErrors.length > 0) {
      console.log(`${pageErrors.length} console errors were reported:`);
      for (const pageError of pageErrors.slice(0, 20)) {
        console.log(`  ${pageError.split('\n')[0]}`);
      }
    }
    process.exitCode =
      result.reportedFailures > 0 || result.initErrors.length > 0 ? 1 : 0;
  } finally {
    if (browser) {
      await browser.disconnect();
    }
    if (chrome) {
      chrome.kill();
      await waitForExit(chrome);
    }
    if (server) {
      server.kill();
    }
    rmSync(userDataDir, { recursive: true, force: true });
  }
}

await main();
