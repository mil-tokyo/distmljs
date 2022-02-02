// mochaをimportすると、webpackがエラーとなる。テスト用html内でCDNからmocha.jsを読み込む。
declare const mocha: any;

mocha.setup('bdd');
/*
テストの追加方法
XXX: テスト対象グループ名、YYY: テスト対象クラス名
./XXX/YYY.test.ts にテストケースを記載
./XXX/index.ts 内に、"import ./YYY.test"を追記
新規グループを追加する場合は下にimportを追記
*/
import './tensor/index';
import './nn/index';
import { initializeNNWebGLContext } from '../tensor';

window.addEventListener('load', async () => {
  await initializeNNWebGLContext();
  mocha.run();
});
