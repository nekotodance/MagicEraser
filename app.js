(() => {

"use strict";


/* =========================================================
   Configuration
========================================================= */


/*
  Carve/LaMa-ONNX

  lama_fp32.onnx

  Fixed input:
    512 x 512

  Inputs:
    image : [1, 3, 512, 512]
    mask  : [1, 1, 512, 512]

  image:
    float32
    0.0 - 1.0

  mask:
    float32
    0.0 / 1.0

  Model output:
    [1, 3, 512, 512]
*/

const MODEL_SIZE = 512;

const MODEL_URL =
  "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx?download=true";


/*
  消去範囲の周囲にどれくらいコンテキストを
  取るか。

  大きすぎるとモデルの512pxに対して
  消去対象が小さくなりすぎる。

  小さすぎると周辺情報が不足する。

  1.5倍程度を基本値にする。
*/

const CONTEXT_SCALE = 1.8;


/*
  Undo最大数
*/

const MAX_HISTORY = 12;


/* =========================================================
   DOM
========================================================= */

const $ =
  id => document.getElementById(id);


const fileInput =
  $("fileInput");

const saveBtn =
  $("saveBtn");

const clearBtn =
  $("clearBtn");

const undoBtn =
  $("undoBtn");

const redoBtn =
  $("redoBtn");

const eraseBtn =
  $("eraseBtn");

const clearMaskBtn =
  $("clearMaskBtn");

const brushSize =
  $("brushSize");

const brushSizeValue =
  $("brushSizeValue");

const workspace =
  $("workspace");

const dropHint =
  $("dropHint");

const canvasWrap =
  $("canvasWrap");

const imageCanvas =
  $("imageCanvas");

const maskCanvas =
  $("maskCanvas");

const cursor =
  $("brushCursor");

const status =
  $("status");

const runtimeStatus =
  $("runtimeStatus");


/* =========================================================
   Canvas contexts
========================================================= */

const ictx =
  imageCanvas.getContext(
    "2d",
    {
      willReadFrequently: true
    }
  );

const mctx =
  maskCanvas.getContext(
    "2d",
    {
      willReadFrequently: true
    }
  );


/* =========================================================
   State
========================================================= */

let imageLoaded = false;

let drawing = false;

let eraseMode = false;

let lastPoint = null;

let undoStack = [];

let redoStack = [];

let currentFileName = "";

/*
  ONNX Runtime session
*/

let lamaSession = null;

let lamaRuntime = null;


/*
  モデルロード中のPromise。

  同時に複数回ロードされるのを防ぐ。
*/

let lamaLoadingPromise = null;


/* =========================================================
   Status
========================================================= */

function setStatus(text) {

  status.textContent = text;

}


function setRuntimeStatus(text) {

  runtimeStatus.textContent = text;

}


/* =========================================================
   Button state
========================================================= */

function updateButtons() {

  undoBtn.disabled =
    undoStack.length === 0;

  redoBtn.disabled =
    redoStack.length === 0;

  eraseBtn.disabled =
    !imageLoaded ||
    !hasMask();

  clearMaskBtn.disabled =
    !imageLoaded ||
    !hasMask();

  saveBtn.disabled =
    !imageLoaded;

  clearBtn.disabled =
    !imageLoaded;

}


/* =========================================================
   History
========================================================= */

function snapshot() {

  return {

    image:
      ictx.getImageData(
        0,
        0,
        imageCanvas.width,
        imageCanvas.height
      ),

    mask:
      mctx.getImageData(
        0,
        0,
        maskCanvas.width,
        maskCanvas.height
      )

  };

}


function restore(state) {

  ictx.putImageData(
    state.image,
    0,
    0
  );

  mctx.putImageData(
    state.mask,
    0,
    0
  );

  updateButtons();

}


function pushUndo() {

  if (!imageLoaded) {
    return;
  }

  undoStack.push(
    snapshot()
  );

  if (
    undoStack.length >
    MAX_HISTORY
  ) {

    undoStack.shift();

  }

  redoStack = [];

  updateButtons();

}


function undo() {

  if (!undoStack.length) {
    return;
  }

  redoStack.push(
    snapshot()
  );

  restore(
    undoStack.pop()
  );

  setStatus(
    "Undo"
  );

}


function redo() {

  if (!redoStack.length) {
    return;
  }

  undoStack.push(
    snapshot()
  );

  restore(
    redoStack.pop()
  );

  setStatus(
    "Redo"
  );

}


/* =========================================================
   Check mask
========================================================= */

function hasMask() {

  if (!imageLoaded) {
    return false;
  }


  const data =
    mctx.getImageData(
      0,
      0,
      maskCanvas.width,
      maskCanvas.height
    ).data;


  for (
    let i = 3;
    i < data.length;
    i += 4
  ) {

    if (data[i] > 0) {
      return true;
    }

  }


  return false;

}


/* =========================================================
   Reset
========================================================= */


/*
  「クリア」

  ・画像
  ・マスク
  ・Undo
  ・Redo
  ・ファイル選択
  ・モデル状態は維持

  を初期状態へ戻す。

  モデル自体は200MB程度あるため、
  クリアのたびに再ロードはしない。
*/

function clearAll() {

  imageLoaded = false;

  drawing = false;

  eraseMode = false;

  lastPoint = null;
  
  currentFileName = "";

  undoStack = [];

  redoStack = [];


  ictx.clearRect(
    0,
    0,
    imageCanvas.width,
    imageCanvas.height
  );

  mctx.clearRect(
    0,
    0,
    maskCanvas.width,
    maskCanvas.height
  );


  imageCanvas.width = 1;
  imageCanvas.height = 1;

  maskCanvas.width = 1;
  maskCanvas.height = 1;


  canvasWrap.hidden = true;

  dropHint.hidden = false;


  fileInput.value = "";


  cursor.style.display =
    "none";


  setStatus(
    "画像を開いてください"
  );


  updateButtons();

}


/* =========================================================
   Fit canvas
========================================================= */

function fitCanvasToViewport() {

  if (!imageLoaded) {
    return;
  }


  const maxW =
    Math.max(
      300,
      workspace.clientWidth - 40
    );


  const maxH =
    Math.max(
      300,
      workspace.clientHeight - 40
    );


  const scale =
    Math.min(
      1,
      maxW / imageCanvas.width,
      maxH / imageCanvas.height
    );


  const w =
    Math.max(
      1,
      Math.round(
        imageCanvas.width * scale
      )
    );


  const h =
    Math.max(
      1,
      Math.round(
        imageCanvas.height * scale
      )
    );


  canvasWrap.style.width =
    w + "px";

  canvasWrap.style.height =
    h + "px";


  imageCanvas.style.width =
    w + "px";

  imageCanvas.style.height =
    h + "px";


  maskCanvas.style.width =
    w + "px";

  maskCanvas.style.height =
    h + "px";

}


/* =========================================================
   Open image
========================================================= */

function openFile(file) {

  if (
    !file ||
    !file.type.startsWith("image/")
  ) {

    setStatus(
      "画像ファイルを選択してください"
    );

    return;

  }
  currentFileName = file.name;


  const url =
    URL.createObjectURL(file);


  const img =
    new Image();


  img.onload = () => {

    imageCanvas.width =
      img.naturalWidth;

    imageCanvas.height =
      img.naturalHeight;


    maskCanvas.width =
      img.naturalWidth;

    maskCanvas.height =
      img.naturalHeight;


    ictx.clearRect(
      0,
      0,
      imageCanvas.width,
      imageCanvas.height
    );


    ictx.drawImage(
      img,
      0,
      0
    );


    mctx.clearRect(
      0,
      0,
      maskCanvas.width,
      maskCanvas.height
    );


    URL.revokeObjectURL(url);


    imageLoaded = true;


    undoStack = [];

    redoStack = [];


    /*
      重要：
      hidden属性だけでなく、
      CSS側でもhiddenを処理している。
    */

    dropHint.hidden = true;

    canvasWrap.hidden = false;


    fitCanvasToViewport();


    setStatus(
      `${img.naturalWidth} × ${img.naturalHeight} : ` + currentFileName
    );


    updateButtons();

  };


  img.onerror = () => {

    URL.revokeObjectURL(url);

    setStatus(
      "画像を読み込めませんでした"
    );

  };


  img.src = url;

}


/* =========================================================
   File input
========================================================= */

fileInput.addEventListener(
  "change",
  e => {

    openFile(
      e.target.files[0]
    );

  }
);


/* =========================================================
   Drag & Drop
========================================================= */

[
  "dragenter",
  "dragover"
].forEach(
  eventName => {

    workspace.addEventListener(
      eventName,
      e => {

        e.preventDefault();

        workspace.classList.add(
          "dragover"
        );

      }
    );

  }
);


[
  "dragleave",
  "drop"
].forEach(
  eventName => {

    workspace.addEventListener(
      eventName,
      e => {

        e.preventDefault();

        workspace.classList.remove(
          "dragover"
        );

      }
    );

  }
);


workspace.addEventListener(
  "drop",
  e => {

    const file =
      [...e.dataTransfer.files]
        .find(
          f =>
            f.type.startsWith("image/")
        );


    if (file) {

      openFile(file);

    }

  }
);


/* =========================================================
   Canvas coordinates
========================================================= */

function canvasPoint(e) {

  const rect =
    imageCanvas.getBoundingClientRect();


  return {

    x:
      (e.clientX - rect.left) *
      imageCanvas.width /
      rect.width,

    y:
      (e.clientY - rect.top) *
      imageCanvas.height /
      rect.height

  };

}


/* =========================================================
   Brush cursor
========================================================= */

function updateCursor(e) {

  if (!imageLoaded) {
    return;
  }


  const rect =
    imageCanvas.getBoundingClientRect();


  const size =
    Number(brushSize.value) *
    rect.width /
    imageCanvas.width;


  cursor.style.width =
    size + "px";

  cursor.style.height =
    size + "px";


  cursor.style.left =
    (e.clientX - rect.left) +
    "px";

  cursor.style.top =
    (e.clientY - rect.top) +
    "px";


  cursor.style.display =
    "block";

}


/* =========================================================
   Paint mask
========================================================= */

function paintAt(
  point,
  erase
) {

  const radius =
    Number(brushSize.value) / 2;


  mctx.save();


  if (erase) {

    mctx.globalCompositeOperation =
      "destination-out";

  } else {

    mctx.globalCompositeOperation =
      "source-over";

  }


  mctx.strokeStyle =
    "rgba(255,55,55,0.72)";

  mctx.fillStyle =
    "rgba(255,55,55,0.72)";


  mctx.lineCap =
    "round";

  mctx.lineJoin =
    "round";


  mctx.lineWidth =
    Number(brushSize.value);


  if (!lastPoint) {

    mctx.beginPath();

    mctx.arc(
      point.x,
      point.y,
      radius,
      0,
      Math.PI * 2
    );

    mctx.fill();

  } else {

    mctx.beginPath();

    mctx.moveTo(
      lastPoint.x,
      lastPoint.y
    );

    mctx.lineTo(
      point.x,
      point.y
    );

    mctx.stroke();

  }


  mctx.restore();

}


/* =========================================================
   Prevent context menu
========================================================= */

imageCanvas.addEventListener(
  "contextmenu",
  e => e.preventDefault()
);

canvasWrap.addEventListener(
  "contextmenu",
  e => e.preventDefault()
);


/* =========================================================
   Pointer events
========================================================= */

canvasWrap.addEventListener(
  "pointermove",
  e => {

    updateCursor(e);


    if (!drawing) {
      return;
    }


    const point =
      canvasPoint(e);


    paintAt(
      point,
      eraseMode
    );


    lastPoint =
      point;


    updateButtons();

  }
);


canvasWrap.addEventListener(
  "pointerenter",
  e => {

    updateCursor(e);

  }
);


canvasWrap.addEventListener(
  "pointerleave",
  () => {

    cursor.style.display =
      "none";

  }
);


canvasWrap.addEventListener(
  "pointerdown",
  e => {

    if (
      !imageLoaded ||
      (
        e.button !== 0 &&
        e.button !== 2
      )
    ) {

      return;

    }


    e.preventDefault();


    pushUndo();


    drawing = true;


    eraseMode =
      e.button === 2;


    lastPoint = null;


    const point =
      canvasPoint(e);


    paintAt(
      point,
      eraseMode
    );


    lastPoint =
      point;

  }
);


window.addEventListener(
  "pointerup",
  () => {

    drawing = false;

    lastPoint = null;

    updateButtons();

  }
);


/* =========================================================
   Brush size
========================================================= */

brushSize.addEventListener(
  "input",
  () => {

    brushSizeValue.textContent =
      brushSize.value + " px";


    if (!imageLoaded) {
      return;
    }


    const rect =
      imageCanvas.getBoundingClientRect();


    const size =
      Number(brushSize.value) *
      rect.width /
      imageCanvas.width;


    cursor.style.width =
      size + "px";

    cursor.style.height =
      size + "px";

  }
);


/* =========================================================
   Clear mask
========================================================= */

clearMaskBtn.addEventListener(
  "click",
  () => {

    if (!hasMask()) {
      return;
    }


    pushUndo();


    mctx.clearRect(
      0,
      0,
      maskCanvas.width,
      maskCanvas.height
    );


    setStatus(
      "消去範囲を解除しました"
    );


    updateButtons();

  }
);


/* =========================================================
   Get mask bounding box
========================================================= */

function getMaskBounds() {

  const width =
    maskCanvas.width;

  const height =
    maskCanvas.height;


  const data =
    mctx.getImageData(
      0,
      0,
      width,
      height
    ).data;


  let minX = width;
  let minY = height;

  let maxX = -1;
  let maxY = -1;


  for (
    let y = 0;
    y < height;
    y++
  ) {

    for (
      let x = 0;
      x < width;
      x++
    ) {

      const alpha =
        data[
          (y * width + x) * 4 + 3
        ];


      if (alpha > 20) {

        if (x < minX) minX = x;

        if (y < minY) minY = y;

        if (x > maxX) maxX = x;

        if (y > maxY) maxY = y;

      }

    }

  }


  if (maxX < 0) {
    return null;
  }


  return {
    minX,
    minY,
    maxX,
    maxY
  };

}


/* =========================================================
   Create square inference region
========================================================= */


/*
  LaMaは512x512固定。

  消去対象だけを切り出すのではなく、
  周辺のコンテキストも含めた正方形領域を作る。

  その正方形を512x512にリサイズして推論する。
*/

function getInferenceRegion(bounds) {

  const imageW =
    imageCanvas.width;

  const imageH =
    imageCanvas.height;


  const maskW =
    bounds.maxX -
    bounds.minX +
    1;

  const maskH =
    bounds.maxY -
    bounds.minY +
    1;


  const context =
    Math.max(
      32,
      Math.round(
        Math.max(maskW, maskH) *
        CONTEXT_SCALE
      )
    );


  let size =
    Math.max(
      maskW,
      maskH,
      context
    );


  /*
    正方形サイズが画像より大きい場合は
    画像サイズまで縮める。
  */

  size =
    Math.min(
      size,
      imageW,
      imageH
    );


  /*
    マスク中心
  */

  const cx =
    (bounds.minX + bounds.maxX) / 2;

  const cy =
    (bounds.minY + bounds.maxY) / 2;


  let x =
    Math.round(
      cx - size / 2
    );

  let y =
    Math.round(
      cy - size / 2
    );


  /*
    画像外には出さない
  */

  x =
    Math.max(
      0,
      Math.min(
        x,
        imageW - size
      )
    );


  y =
    Math.max(
      0,
      Math.min(
        y,
        imageH - size
      )
    );


  return {

    x,
    y,

    size

  };

}


/* =========================================================
   Create model input
========================================================= */

function createModelInput(region) {

  /*
    一時Canvas
  */

  const imageCanvas512 =
    document.createElement(
      "canvas"
    );

  imageCanvas512.width =
    MODEL_SIZE;

  imageCanvas512.height =
    MODEL_SIZE;


  const maskCanvas512 =
    document.createElement(
      "canvas"
    );

  maskCanvas512.width =
    MODEL_SIZE;

  maskCanvas512.height =
    MODEL_SIZE;


  const imageCtx =
    imageCanvas512.getContext(
      "2d",
      {
        willReadFrequently: true
      }
    );


  const maskCtx =
    maskCanvas512.getContext(
      "2d",
      {
        willReadFrequently: true
      }
    );


  /*
    元画像を512x512へ
  */

  imageCtx.drawImage(
    imageCanvas,

    region.x,
    region.y,
    region.size,
    region.size,

    0,
    0,
    MODEL_SIZE,
    MODEL_SIZE
  );


  /*
    マスクを512x512へ。

    NEAREST相当になるように
    imageSmoothingEnabledをOFF。
  */

  maskCtx.imageSmoothingEnabled =
    false;


  maskCtx.drawImage(
    maskCanvas,

    region.x,
    region.y,
    region.size,
    region.size,

    0,
    0,
    MODEL_SIZE,
    MODEL_SIZE
  );


  const imageData =
    imageCtx.getImageData(
      0,
      0,
      MODEL_SIZE,
      MODEL_SIZE
    );


  const maskData =
    maskCtx.getImageData(
      0,
      0,
      MODEL_SIZE,
      MODEL_SIZE
    );


  /*
    ONNX input

    image:
      [1,3,512,512]

    mask:
      [1,1,512,512]
  */

  const imageTensorData =
    new Float32Array(
      3 *
      MODEL_SIZE *
      MODEL_SIZE
    );


  const maskTensorData =
    new Float32Array(
      MODEL_SIZE *
      MODEL_SIZE
    );


  const planeSize =
    MODEL_SIZE *
    MODEL_SIZE;


  for (
    let y = 0;
    y < MODEL_SIZE;
    y++
  ) {

    for (
      let x = 0;
      x < MODEL_SIZE;
      x++
    ) {

      const pixel =
        (
          y *
          MODEL_SIZE +
          x
        ) * 4;


      const p =
        y *
        MODEL_SIZE +
        x;


      /*
        RGB -> CHW
      */

      imageTensorData[
        p
      ] =
        imageData.data[
          pixel
        ] / 255;


      imageTensorData[
        planeSize + p
      ] =
        imageData.data[
          pixel + 1
        ] / 255;


      imageTensorData[
        planeSize * 2 + p
      ] =
        imageData.data[
          pixel + 2
        ] / 255;


      /*
        Alpha > 20をマスクとして扱う。

        赤色表示部分のalphaを
        0/1に変換する。
      */

      maskTensorData[p] =
        maskData.data[
          pixel + 3
        ] > 20
          ? 1.0
          : 0.0;

    }

  }


  return {

    image:
      new ort.Tensor(
        "float32",
        imageTensorData,
        [
          1,
          3,
          MODEL_SIZE,
          MODEL_SIZE
        ]
      ),

    mask:
      new ort.Tensor(
        "float32",
        maskTensorData,
        [
          1,
          1,
          MODEL_SIZE,
          MODEL_SIZE
        ]
      ),

    region

  };

}


/* =========================================================
   Load LaMa
========================================================= */

async function loadLaMa() {

  if (lamaSession) {
    return lamaSession;
  }


  if (lamaLoadingPromise) {
    return lamaLoadingPromise;
  }


  lamaLoadingPromise =
    (async () => {

      setStatus(
        "LaMaモデルを読み込んでいます…"
      );


      setRuntimeStatus(
        "LaMa: モデル読み込み中"
      );


      /*
        WASMファイルの場所
      */

      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";


      /*
        WebGPUが利用可能なら優先。

        利用できない場合はWASM。
      */

      const webgpuAvailable =
        "gpu" in navigator;


      if (webgpuAvailable) {

        try {

          setRuntimeStatus(
            "LaMa: WebGPU初期化中"
          );


          lamaSession =
            await ort.InferenceSession.create(
              MODEL_URL,
              {
                executionProviders: [
                  "webgpu"
                ],

                graphOptimizationLevel:
                  "all"
              }
            );


          lamaRuntime =
            "WebGPU";


          setRuntimeStatus(
            "LaMa: WebGPU"
          );


          return lamaSession;

        } catch (error) {

          console.warn(
            "WebGPU LaMa initialization failed:",
            error
          );


          lamaSession = null;

          setRuntimeStatus(
            "LaMa: WebGPU失敗 → WASM"
          );

        }

      }


      /*
        CPU / WASM fallback
      */

      lamaSession =
        await ort.InferenceSession.create(
          MODEL_URL,
          {
            executionProviders: [
              "wasm"
            ],

            graphOptimizationLevel:
              "all"
          }
        );


      lamaRuntime =
        "WASM";


      setRuntimeStatus(
        "LaMa: WASM"
      );


      return lamaSession;

    })();


  try {

    return await lamaLoadingPromise;

  } catch (error) {

    lamaLoadingPromise = null;

    lamaSession = null;

    throw error;

  }

}


/* =========================================================
   Run LaMa
========================================================= */

async function runLaMa() {

  const bounds =
    getMaskBounds();


  if (!bounds) {

    throw new Error(
      "Mask is empty."
    );

  }


  const region =
    getInferenceRegion(
      bounds
    );


  /*
    モデル入力を生成
  */

  const input =
    createModelInput(
      region
    );


  /*
    入力名はモデルカードで
    image / mask とされている。

    念のため実際のモデル入力名を確認して
    マッピングする。
  */

  const inputs =
    lamaSession.inputNames;


  let imageName =
    "image";

  let maskName =
    "mask";


  /*
    もし名前が違うモデルだった場合にも
    最低限対応する。
  */

  if (!inputs.includes("image")) {

    imageName =
      inputs[0];

  }


  if (!inputs.includes("mask")) {

    maskName =
      inputs[1];

  }


  /*
    推論
  */

  const results =
    await lamaSession.run({

      [imageName]:
        input.image,

      [maskName]:
        input.mask

    });


  /*
    最初のoutput
  */

  const outputName =
    lamaSession.outputNames[0];


  const output =
    results[outputName];


  /*
    出力は
      [1,3,512,512]

    のCHW。
  */

  const outputData =
    output.data;


  const outputCanvas =
    document.createElement(
      "canvas"
    );


  outputCanvas.width =
    MODEL_SIZE;

  outputCanvas.height =
    MODEL_SIZE;


  const outputCtx =
    outputCanvas.getContext(
      "2d"
    );


  const resultImage =
    outputCtx.createImageData(
      MODEL_SIZE,
      MODEL_SIZE
    );


  const planeSize =
    MODEL_SIZE *
    MODEL_SIZE;


  /*
    モデルによって

      0～1

    または

      0～255

    の出力になる可能性がある。

    Carve版はuint8変換を前提とした
    出力になっているため、
    最大値を見て判定する。
  */

  let maxValue = 0;


  /*
    全値を調べる。

    512x512x3なので十分軽い。
  */

  for (
    let i = 0;
    i < outputData.length;
    i++
  ) {

    if (
      outputData[i] >
      maxValue
    ) {

      maxValue =
        outputData[i];

    }

  }


  const scale =
    maxValue <= 2
      ? 255
      : 1;


  for (
    let y = 0;
    y < MODEL_SIZE;
    y++
  ) {

    for (
      let x = 0;
      x < MODEL_SIZE;
      x++
    ) {

      const p =
        y *
        MODEL_SIZE +
        x;


      const dst =
        p * 4;


      resultImage.data[
        dst
      ] =
        clampByte(
          outputData[p] *
          scale
        );


      resultImage.data[
        dst + 1
      ] =
        clampByte(
          outputData[
            planeSize + p
          ] *
          scale
        );


      resultImage.data[
        dst + 2
      ] =
        clampByte(
          outputData[
            planeSize * 2 + p
          ] *
          scale
        );


      resultImage.data[
        dst + 3
      ] =
        255;

    }

  }


  outputCtx.putImageData(
    resultImage,
    0,
    0
  );


  /*
    元画像に戻す。

    LaMa結果全体を置き換えるのではなく、
    マスク部分だけを適用する。
  */

  const originalRegion =
    document.createElement(
      "canvas"
    );


  originalRegion.width =
    region.size;

  originalRegion.height =
    region.size;


  const originalCtx =
    originalRegion.getContext(
      "2d"
    );


  originalCtx.drawImage(
    imageCanvas,

    region.x,
    region.y,
    region.size,
    region.size,

    0,
    0,
    region.size,
    region.size
  );


  /*
    LaMa結果を元の領域サイズへ戻す
  */

  const resizedResult =
    document.createElement(
      "canvas"
    );


  resizedResult.width =
    region.size;

  resizedResult.height =
    region.size;


  const resizedCtx =
    resizedResult.getContext(
      "2d"
    );


  resizedCtx.drawImage(
    outputCanvas,

    0,
    0,
    MODEL_SIZE,
    MODEL_SIZE,

    0,
    0,
    region.size,
    region.size
  );


  /*
    元のマスクも領域サイズにする
  */

  const regionMask =
    document.createElement(
      "canvas"
    );


  regionMask.width =
    region.size;

  regionMask.height =
    region.size;


  const regionMaskCtx =
    regionMask.getContext(
      "2d"
    );


  regionMaskCtx.imageSmoothingEnabled =
    false;


  regionMaskCtx.drawImage(
    maskCanvas,

    region.x,
    region.y,
    region.size,
    region.size,

    0,
    0,
    region.size,
    region.size
  );


  /*
    LaMa結果を
    マスク部分だけ残す。
  */

  resizedCtx.globalCompositeOperation =
    "destination-in";


  resizedCtx.drawImage(
    regionMask,
    0,
    0
  );


  /*
    元画像の該当部分に
    LaMa結果を重ねる。
  */

  ictx.drawImage(
    resizedResult,
    region.x,
    region.y
  );


  /*
    マスクを消す。
  */

  mctx.clearRect(
    0,
    0,
    maskCanvas.width,
    maskCanvas.height
  );

}


/* =========================================================
   Clamp
========================================================= */

function clampByte(value) {

  if (value < 0) {
    return 0;
  }

  if (value > 255) {
    return 255;
  }

  return Math.round(value);

}


/* =========================================================
   Erase button
========================================================= */

eraseBtn.addEventListener(
  "click",
  async () => {

    if (
      !imageLoaded ||
      !hasMask()
    ) {

      return;

    }


    /*
      現在状態をUndoへ保存
    */

    pushUndo();


    /*
      推論開始
    */

    eraseBtn.disabled = true;

    clearBtn.disabled = true;

    clearMaskBtn.disabled = true;


    try {

      setStatus(
        "LaMaを準備しています…"
      );


      await loadLaMa();


      setStatus(
        `LaMaで補完中 (${lamaRuntime})…`
      );


      /*
        UIを一度更新してから
        重い推論を開始する。
      */

      await new Promise(
        resolve =>
          requestAnimationFrame(
            resolve
          )
      );


      await runLaMa();


      setStatus(
        "消去完了"
      );


    } catch (error) {

      console.error(
        "LaMa inference failed:",
        error
      );


      /*
        今回の変更をUndo。

        pushUndo()で直前状態が
        undoStackに入っているため、
        そこから戻す。
      */

      if (undoStack.length) {

        restore(
          undoStack.pop()
        );

      }


      setStatus(
        "LaMa処理に失敗しました。コンソールを確認してください"
      );


      alert(
        "LaMaの推論に失敗しました。\n\n" +
        error.message
      );

    }


    updateButtons();

  }
);


/* =========================================================
   Save
========================================================= */

saveBtn.addEventListener(
  "click",
  () => {

    if (!imageLoaded) {
      return;
    }


    const a =
      document.createElement(
        "a"
      );


    a.download =
      currentFileName || "image_cleaned.png";


    a.href =
      imageCanvas.toDataURL(
        "image/png"
      );


    a.click();


    setStatus(
      currentFileName + "を保存しました"
    );

  }
);


/* =========================================================
   Clear
========================================================= */

clearBtn.addEventListener(
  "click",
  () => {

    clearAll();

  }
);


/* =========================================================
   Undo / Redo
========================================================= */

undoBtn.addEventListener(
  "click",
  undo
);


redoBtn.addEventListener(
  "click",
  redo
);


/* =========================================================
   Keyboard shortcuts
========================================================= */

window.addEventListener(
  "keydown",
  e => {

    const mod =
      e.ctrlKey ||
      e.metaKey;


    /*
      Ctrl + Z
    */

    if (
      mod &&
      e.key.toLowerCase() === "z" &&
      !e.shiftKey
    ) {

      e.preventDefault();

      undo();

    }


    /*
      Ctrl + Y
      Ctrl + Shift + Z
    */

    else if (
      mod &&
      (
        e.key.toLowerCase() === "y" ||
        (
          e.key.toLowerCase() === "z" &&
          e.shiftKey
        )
      )
    ) {

      e.preventDefault();

      redo();

    }

  }
);


/* =========================================================
   Resize
========================================================= */

window.addEventListener(
  "resize",
  fitCanvasToViewport
);


/* =========================================================
   Initialization
========================================================= */

updateButtons();

})();
