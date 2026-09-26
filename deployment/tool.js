const API_URL = "https://mice-shelter-lifetime-alfred.trycloudflare.com";

const fileInput = document.getElementById('file-input');
const runBtn = document.getElementById('run-btn');
const boxOriginal = document.getElementById('box-original');
const boxRecon = document.getElementById('box-recon');
const boxProtected = document.getElementById('box-protected');
const imgOriginal = document.getElementById('img-original');
const imgRecon = document.getElementById('img-recon');
const imgProtected = document.getElementById('img-protected');
const statusEl = document.getElementById('status');

const stat = {
  mean: document.getElementById('stat-mean'),
  std: document.getElementById('stat-std'),
  shape: document.getElementById('stat-shape'),
  mse: document.getElementById('stat-mse'),
  dist: document.getElementById('stat-dist'),
};

const evalStat = {
  latent: document.getElementById('eval-latent'),
  linf: document.getElementById('eval-linf'),
  budget: document.getElementById('eval-budget'),
  psnr: document.getElementById('eval-psnr'),
  mirage: document.getElementById('eval-mirage'),
};

const lossCanvas = document.getElementById('loss-canvas');
const mirageCanvas = document.getElementById('mirage-canvas');
const ctx = lossCanvas.getContext('2d');
const mirageCtx = mirageCanvas.getContext('2d');
let currentFile = null;
const attackOption = document.getElementById('attack-option');
const evaluationOption = document.getElementById('evaluation-option');
const attackSection = document.getElementById('attack-section');
const evaluationSection = document.getElementById('evaluation-section');
const evalOriginal = document.getElementById('eval-original');
const evalProtected = document.getElementById('eval-protected');
const evalAi = document.getElementById('eval-ai-reconstruction');
const downStat = {
  model: document.getElementById('down-model'),
  psnr: document.getElementById('down-psnr'),
  mse: document.getElementById('down-mse'),
  protectedPsnr: document.getElementById('down-protected-psnr'),
  protectedMse: document.getElementById('down-protected-mse'),
};

attackOption.addEventListener('click', () => {
  attackOption.classList.add('active');
  evaluationOption.classList.remove('active');
  attackSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

evaluationOption.addEventListener('click', () => {
  evaluationOption.classList.add('active');
  attackOption.classList.remove('active');
  evaluationSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
});


fileInput.addEventListener('change', function (event) {
  const file = event.target.files[0];
  if (!file) return;
  currentFile = file;
  const url = URL.createObjectURL(file);
  imgOriginal.src = url;
  imgOriginal.style.display = 'block';
  boxOriginal.querySelector('span').style.display = 'none';
  resetBox(boxRecon, imgRecon, 'Waiting for run');
  resetBox(boxProtected, imgProtected, 'Waiting for run');
  resetStats();
  resetEvaluation();
  resetDownstream();
  drawCurve(ctx, lossCanvas, []);
  drawCurve(mirageCtx, mirageCanvas, []);
  runBtn.disabled = false;
  statusEl.textContent = 'Status: Loaded ' + file.name;
});

runBtn.addEventListener('click', async function () {
  if (!currentFile) return;
  runBtn.disabled = true;
  statusEl.textContent = 'Status: Running combined VAE → PGD → MIRAGE attack...';
  try {
    const result = await runPipeline(currentFile);

    imgRecon.src = result.reconstructionUrl;
    imgRecon.style.display = 'block';
    boxRecon.querySelector('span').style.display = 'none';

    imgProtected.src = result.protectedUrl;
    imgProtected.style.display = 'block';
    boxProtected.querySelector('span').style.display = 'none';

    stat.mean.textContent = result.latentMean;
    stat.std.textContent = result.latentStd;
    stat.shape.textContent = result.latentShape;
    stat.mse.textContent = result.reconstructionMse;
    stat.dist.textContent = result.finalLatentDistance;

    drawCurve(ctx, lossCanvas, result.pgdLossCurve || result.lossCurve || []);
    drawCurve(mirageCtx, mirageCanvas, result.mirageLossCurve || []);

    evalOriginal.src = result.originalUrl || imgOriginal.src;
    evalProtected.src = result.protectedUrl || imgProtected.src;
    evalAi.src = result.aiReconstructionUrl || '';
    [evalOriginal, evalProtected, evalAi].forEach(img => { img.style.display = img.src ? 'block' : 'none'; });

    const e = result.evaluation || {};
    evalStat.latent.textContent = fmt(e.latentDistance);
    evalStat.linf.textContent = fmt(e.perturbationLinf);
    evalStat.budget.textContent = e.budgetUtilization == null ? '-' : (e.budgetUtilization * 100).toFixed(1) + '%';
    evalStat.psnr.textContent = e.protectedPsnrDb == null ? '-' : e.protectedPsnrDb.toFixed(2) + ' dB';
    evalStat.mirage.textContent = e.mirageScoreChange == null ? '-' : fmt(e.mirageScoreChange);

    const d = result.downstreamEvaluation || {};
    downStat.model.textContent = d.method || '-';
    downStat.psnr.textContent = d.generatedVsOriginalPsnrDb == null ? '-' : Number(d.generatedVsOriginalPsnrDb).toFixed(2) + ' dB';
    downStat.mse.textContent = fmt(d.generatedVsOriginalMse);
    downStat.protectedPsnr.textContent = d.generatedVsProtectedPsnrDb == null ? '-' : Number(d.generatedVsProtectedPsnrDb).toFixed(2) + ' dB';
    downStat.protectedMse.textContent = fmt(d.generatedVsProtectedMse);

    statusEl.textContent = 'Status: Combined attack complete — backend evaluation complete';
  } catch (error) {
    statusEl.textContent = 'Status: Error — ' + error.message;
  } finally {
    runBtn.disabled = false;
  }
});

async function runPipeline(imageFile) {
  const form = new FormData();
  form.append('file', imageFile);
  const response = await fetch(`${API_URL}/protect`, { method: 'POST', body: form });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const error = await response.json();
      message = error.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  return await response.json();
}

function resetBox(box, img, message) {
  img.style.display = 'none';
  img.removeAttribute('src');
  const span = box.querySelector('span');
  span.textContent = message;
  span.style.display = 'block';
}

function resetStats() {
  Object.values(stat).forEach(el => el.textContent = '-');
}

function resetEvaluation() {
  Object.values(evalStat).forEach(el => el.textContent = '-');
  [evalOriginal, evalProtected, evalAi].forEach(img => { img.removeAttribute('src'); img.style.display = 'none'; });
}

function resetDownstream() {
  Object.values(downStat).forEach(el => el.textContent = '-');
}

function fmt(value) {
  if (value == null || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(6);
}

function drawCurve(context, canvas, values) {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, rect.width);
  canvas.height = 220;
  const w = canvas.width;
  const h = canvas.height;
  context.clearRect(0, 0, w, h);

  if (!values || values.length === 0) {
    context.fillStyle = '#999';
    context.font = '13px Arial';
    context.textAlign = 'center';
    context.fillText('No data yet', w / 2, h / 2);
    return;
  }

  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = (max - min) || 1;
  const padding = 16;
  const points = values.map((v, i) => {
    const x = padding + (i / Math.max(values.length - 1, 1)) * (w - padding * 2);
    const y = h - padding - ((v - min) / range) * (h - padding * 2);
    return [x, y];
  });

  context.beginPath();
  points.forEach((p, i) => i === 0 ? context.moveTo(p[0], p[1]) : context.lineTo(p[0], p[1]));
  context.strokeStyle = '#2c3e50';
  context.lineWidth = 2;
  context.stroke();
}

drawCurve(ctx, lossCanvas, []);
drawCurve(mirageCtx, mirageCanvas, []);
