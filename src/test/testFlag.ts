export type TestTarget = 'heavy' | 'webgl' | 'webgpu';
export const AllTestTargets: TestTarget[] = ['heavy', 'webgl', 'webgpu'];
/*
一部環境でしか動作しないテストや、時間がかかるテストの実行可否を手動設定する機構
*/

export type TestFlag = { [K in TestTarget]?: boolean };

function loadTestFlag(): TestFlag {
  const params = new URLSearchParams(window.location.search);
  const target = params.get('target');
  if (target != null) {
    const map: Record<string, boolean> = {};
    for (const key of target.split(',')) {
      map[key] = true;
    }
    return map as unknown as TestFlag;
  } else {
    return { webgl: true };
  }
}

export const testFlag = loadTestFlag();

export function saveTestFlagAndReload(flag: TestFlag): void {
  const targets: string[] = [];
  for (const [k, v] of Object.entries(flag)) {
    if (v) {
      targets.push(k);
    }
  }
  const params = new URLSearchParams(window.location.search);
  params.set('target', targets.join(','));
  window.location.search = params.toString();
}
