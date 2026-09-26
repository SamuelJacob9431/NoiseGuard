/* Set this to your Cloudflare URL, without /protect. */
const API_URL = "YOUR_CLOUDFLARE_URL";

const fileInput = document.getElementById("fileInput");
const runBtn = document.getElementById("runBtn");
const fileName = document.getElementById("fileName");

const originalImg = document.getElementById("originalImg");
const protectedImg = document.getElementById("protectedImg");
const reconImg = document.getElementById("reconImg");

const originalBox = document.getElementById("originalBox");
const protectedBox = document.getElementById("protectedBox");
const reconBox = document.getElementById("reconBox");

const backendStatus = document.getElementById("backendStatus");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");

const canvas = document.getElementById("lossCanvas");
const ctx = canvas.getContext("2d");

const stats = {
  mean: document.getElementById("mean"),
  std: document.getElementById("std"),
  shape: document.getElementById("shape"),
  mse: document.getElementById("mse"),
  dist: document.getElementById("dist")
};

let currentFile = null;
let objectUrl = null;

function setStatus(text, type = "") {
  statusText.textContent = text;
  statusPill.className = "pill" + (type ? " " + type : "");
}

function resetBox(box, img, message) {
  img.style.display = "none";
  img.removeAttribute("src");

  box.querySelector(".empty").textContent = message;
  box.querySelector(".empty").style.display = "block";
}

function resetStats() {
  Object.values(stats).forEach((element) => {
    element.textContent = "â€”";
  });
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];

  if (!file) return;

  currentFile = file;
  fileName.textContent = file.name;

  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }

  objectUrl = URL.createObjectURL(file);

  originalImg.src = objectUrl;
  originalImg.style.display = "block";
  originalBox.querySelector(".empty").style.display = "none";

  resetBox(protectedBox, protectedImg, "Waiting for protection");
  resetBox(reconBox, reconImg, "Waiting for protection");

  resetStats();
  drawLossCurve([]);

  runBtn.disabled = false;

  setStatus("Ready");

  backendStatus.textContent =
    API_URL === "YOUR_CLOUDFLARE_URL"
      ? "Add Cloudflare URL"
      : "Configured";
});

runBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  runBtn.disabled = true;

  setStatus("Running", "running");
  backendStatus.textContent = "Sending image to backendâ€¦";

  try {
    const result = await runPipeline(currentFile);

    if (!result || result.success !== true) {
      throw new Error("Backend returned an invalid response.");
    }

    protectedImg.src = result.protectedUrl;
    protectedImg.style.display = "block";
    protectedBox.querySelector(".empty").style.display = "none";

    reconImg.src = result.reconstructionUrl;
    reconImg.style.display = "block";
    reconBox.querySelector(".empty").style.display = "none";

    stats.mean.textContent = formatValue(result.latentMean);
    stats.std.textContent = formatValue(result.latentStd);
    stats.shape.textContent = formatValue(result.latentShape);
    stats.mse.textContent = formatValue(result.reconstructionMse);
    stats.dist.textContent = formatNumber(
      result.maxPerturbation ?? result.finalLatentDistance
    );

    drawLossCurve(result.lossCurve || []);

    backendStatus.textContent = "Protection complete";
    setStatus("Complete", "success");
  } catch (error) {
    console.error(error);

    backendStatus.textContent = error.message;
    setStatus("Failed", "error");

    alert(
      "NoiseGuard could not process the image.\n\n" +
      error.message
    );
  } finally {
    runBtn.disabled = false;
  }
});

async function runPipeline(imageFile) {
  if (!API_URL || API_URL === "YOUR_CLOUDFLARE_URL") {
    throw new Error(
      "Set API_URL to your Cloudflare backend URL first."
    );
  }

  const form = new FormData();
  form.append("file", imageFile);

  const response = await fetch(`${API_URL}/protect`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    let message = `Backend error (${response.status})`;

    try {
      const body = await response.json();

      if (body.detail) {
        message += `: ${body.detail}`;
      }
    } catch (_) {}

    throw new Error(message);
  }

  return await response.json();
}

function formatValue(value) {
  return value === null ||
    value === undefined ||
    value === ""
    ? "â€”"
    : String(value);
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") {
    return "â€”";
  }

  const number = Number(value);

  return Number.isFinite(number)
    ? number.toFixed(6)
    : String(value);
}

function drawLossCurve(values) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = rect.width;
  const height = rect.height;

  ctx.clearRect(0, 0, width, height);

  if (!values.length) {
    ctx.fillStyle = "#657083";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";

    ctx.fillText(
      "Run protection to view the PGD loss curve",
      width / 2,
      height / 2
    );

    return;
  }

  const vals = values
    .map(Number)
    .filter(Number.isFinite);

  if (!vals.length) return;

  const max = Math.max(...vals);
  const min = Math.min(...vals);
  const range = (max - min) || 1;

  const left = 42;
  const right = 16;
  const top = 18;
  const bottom = 28;

  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  ctx.strokeStyle = "#242b36";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 4; i++) {
    const y = top + (i / 4) * plotHeight;

    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - right, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#657083";
  ctx.font = "10px system-ui";
  ctx.textAlign = "right";

  ctx.fillText(max.toFixed(3), left - 7, top + 3);
  ctx.fillText(min.toFixed(3), left - 7, height - bottom);

  ctx.strokeStyle = "#929cff";
  ctx.lineWidth = 2;
  ctx.beginPath();

  vals.forEach((value, index) => {
    const x =
      left +
      (index / Math.max(vals.length - 1, 1)) *
        plotWidth;

    const y =
      top +
      (1 - (value - min) / range) *
        plotHeight;

    if (index) {
      ctx.lineTo(x, y);
    } else {
      ctx.moveTo(x, y);
    }
  });

  ctx.stroke();

  ctx.fillStyle = "#657083";
  ctx.textAlign = "left";
  ctx.fillText("Step 1", left, height - 8);

  ctx.textAlign = "right";
  ctx.fillText(
    `Step ${vals.length}`,
    width - right,
    height - 8
  );
}

drawLossCurve([]);

window.addEventListener("resize", () => {
  drawLossCurve([]);
});