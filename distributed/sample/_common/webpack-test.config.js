module.exports = {
  mode: 'development',
  entry: './src/index_test.ts',

  output: {
    filename: 'index_test.js',
    path: __dirname + '/public/static',
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
