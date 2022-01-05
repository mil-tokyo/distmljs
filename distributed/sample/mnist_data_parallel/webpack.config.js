module.exports = {
  mode: 'development',
  entry: './src/index.ts',

  output: {
    filename: 'index.js',
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
