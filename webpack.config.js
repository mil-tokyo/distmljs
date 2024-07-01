module.exports = {
  mode: 'development',
  entry: './src/index.ts',

  output: {
    filename: 'distmljs.js',
    path: __dirname + '/webpack',
    library: 'distmljs',
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
