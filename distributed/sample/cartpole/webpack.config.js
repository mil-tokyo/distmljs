module.exports = {
  mode: 'development',
  entry: './src/index_sync.ts',

  output: {
    filename: 'index_sync.js',
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
