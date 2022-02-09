// compiles /shader/webgpu/*.glsl into src/tensor/webgpu/shaders.ts

const glslang = require('@webgpu/glslang')();

const fs = require('fs');

let shaderList =
  'export const webgpuShaders: {[index:string]: Uint32Array} = {\n';

// TODO: support custom/autogen

// recursively search *.glsl in this directory
const sourcesDir = __dirname + '/../shader/webgpu';

const namePattern = /^.*\.glsl$/;
const allFiles = [];
const appendFilesInDirectory = (dir) => {
  const files = fs.readdirSync(dir);
  files.forEach((file) => {
    const path = `${dir}/${file}`;
    const stat = fs.statSync(path);
    if (stat.isDirectory()) {
      appendFilesInDirectory(path);
    } else if (stat.isFile() && namePattern.test(file)) {
      allFiles.push(path);
    }
  });
};
appendFilesInDirectory(sourcesDir);

allFiles.forEach((file) => {
  const basename = file.split('/').pop().split('.')[0];
  console.log(`${file} => ${basename}`);
  const shaderSource = fs.readFileSync(file, {
    encoding: 'utf-8',
  });
  const glslShader = glslang.compileGLSL(shaderSource, 'compute');
  // shader name = basename of file
  // TODO: more compressed format
  shaderList += `${basename}: new Uint32Array([${Array.from(
    glslShader
  ).toString()}]),\n`;
});

shaderList += '};';

fs.writeFileSync(`${__dirname}/../src/tensor/webgpu/shaders.ts`, shaderList);
