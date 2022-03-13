import { startTraining } from './train';
import { checkGrad } from './checkGrad';

function sethandler(name: string, method: Function) {
  const elem = document.getElementById(name);
  if (elem) {
    elem.onclick = () => method();
  }
}

window.addEventListener('load', () => {
  sethandler('start-training', startTraining);
  sethandler('check-grad', checkGrad);
});
