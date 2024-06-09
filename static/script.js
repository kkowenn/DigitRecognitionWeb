var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
var isDrawing = false;

ctx.lineWidth = 35;
ctx.strokeStyle = "black";

canvas.width = 280;
canvas.height = 280;

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener("mousedown", function (e) {
  isDrawing = true;
  ctx.lineWidth = 20;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mousemove", function (e) {
  if (isDrawing) {
    ctx.lineWidth = 35;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  }
});

canvas.addEventListener("mouseup", function () {
  isDrawing = false;
});

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerText = "None"; // Reset predicted digit
}

function predictDigit() {
  var dataURL = canvas.toDataURL("image/png");
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: dataURL }),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("result").innerText = data.digit;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
