module.exports = {
  mode: 'development',
  entry: {
    async: './src/index_async.ts',
    visualize: './src/index_visualize.ts',
  },

  output: {
    filename: 'index_[name].js',
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
