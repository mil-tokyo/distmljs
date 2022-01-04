module.exports = {
  mode: 'development',
  entry: './src/test/entry.ts',

  output: {
    filename: 'kakiage-test.js',
    path: __dirname + '/test',
  },

  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader',
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
};
