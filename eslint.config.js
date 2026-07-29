// ESLint flat config (ESLint 9 以降の形式)
const js = require('@eslint/js');
const tseslint = require('typescript-eslint');
const prettier = require('eslint-config-prettier');

module.exports = tseslint.config(
  {
    ignores: [
      // don't ever lint node_modules
      'node_modules/**',
      // don't lint build output (make sure it's set to your correct build folder name)
      'dist/**',
      'webpack/**',
      // test bundle
      'test/distmljs-test.js',
      // config files
      '*.config.js',
      '.*.js',
      'eslint.config.js',
      // sample does not belong to source
      'sample/**',
      'distributed/**',
      // tools are plain JavaScript
      'tools/**',
    ],
  },
  js.configs.recommended,
  tseslint.configs.recommended,
  {
    files: ['**/*.ts'],
    languageOptions: {
      parserOptions: {
        project: './tsconfig.json',
        tsconfigRootDir: __dirname,
      },
    },
    rules: {
      '@typescript-eslint/no-floating-promises': 'error',
      // typescript-eslint 8のrecommendedではerrorだが、ESLint 8時代の設定と
      // 同じくwarnとして扱う
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  },
  prettier
);
