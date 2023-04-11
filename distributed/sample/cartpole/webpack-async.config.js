module.exports = {
  mode: 'development',
  entry: './src/index_async.ts',

  output: {
    filename: 'index_async.js',
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
