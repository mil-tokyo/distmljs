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
import { initializeNNWebGLContext, initializeNNWebGPUContext } from '../tensor';
import {
  AllTestTargets,
  saveTestFlagAndReload,
  testFlag,
  TestFlag,
  TestTarget,
} from './testFlag';

function makeTargetList() {
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  const tl = document.getElementById('target-list')!;
  const currentTarget = testFlag;
  for (const name of AllTestTargets) {
    const label = document.createElement('label');
    const check = document.createElement('input');
    check.type = 'checkbox';
    check.name = name;
    if (currentTarget[name]) {
      check.checked = true;
    }
    label.appendChild(check);
    label.appendChild(document.createTextNode(name));
    tl.appendChild(label);
  }
  const submit = document.createElement('button');
  submit.type = 'button';
  submit.onclick = () => {
    const flag: TestFlag = {};
    const qs = document.querySelectorAll('#target-list input:checked');
    for (let i = 0; i < qs.length; i++) {
      const check = qs[i] as HTMLInputElement;
      flag[check.name as TestTarget] = true;
    }
    saveTestFlagAndReload(flag);
  };
  submit.innerText = 'Select and reload';
  tl.appendChild(submit);
}

window.addEventListener('load', async () => {
  makeTargetList();
  if (testFlag.webgl) {
    try {
      await initializeNNWebGLContext();
    } catch (error) {
      alert(
        `Failed to initialize WebGL. Uncheck webgl in target selection. ${error}`
      );
    }
  }
  if (testFlag.webgpu) {
    try {
      await initializeNNWebGPUContext();
    } catch (error) {
      alert(
        `Failed to initialize WebGPU. Uncheck webgpu in target selection. ${error}`
      );
    }
  }
  mocha.run();
});
