module.exports = {
  mode: 'development',
  entry: './src/index.ts',

  output: {
    filename: 'kakiage.js',
    path: __dirname + '/webpack',
    library: 'kakiage',
    libraryTarget: 'var',
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
