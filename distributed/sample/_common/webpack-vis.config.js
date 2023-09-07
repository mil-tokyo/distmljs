module.exports = {
  mode: 'development',
  entry: './src/index_visualize.ts',

  output: {
    filename: 'index_visualize.js',
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
