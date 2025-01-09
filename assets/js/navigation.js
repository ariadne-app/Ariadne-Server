let point1 = document.getElementById('point1');
let point2 = document.getElementById('point2');
let canvas = document.getElementById('line-canvas');
let container = document.getElementById('canvas-container');
let ctx = canvas.getContext('2d');
let updateInProgress = false;
let updateQueued = false;
let lastPathPoints = [];
let drawCurve = drawPoints;


function adjustCanvasSize() {
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
}

function initialSetup() {
    adjustCanvasSize();
    updatePoints(); // Load the initial path
}

document.addEventListener('DOMContentLoaded', initialSetup);

let isDragging = false;
let activePoint = null;

function movePoint(event, point) {
    let rect = container.getBoundingClientRect();
    let newX = event.clientX - rect.left - point.offsetWidth / 2;
    let newY = event.clientY - rect.top - point.offsetHeight / 2;

    newX = Math.max(0, Math.min(newX, rect.width - point.offsetWidth));
    newY = Math.max(0, Math.min(newY, rect.height - point.offsetHeight));

    point.style.left = `${newX}px`;
    point.style.top = `${newY}px`;

    throttledUpdatePoints();
}

[point1, point2].forEach(point => {
    point.addEventListener('mousedown', (e) => {
        isDragging = true;
        activePoint = point;

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    });
});

function onMouseMove(event) {
    if (isDragging && activePoint) {
        movePoint(event, activePoint);
    }
}

function onMouseUp() {
    isDragging = false;
    activePoint = null;

    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);

    updatePoints();
}

function updatePoints() {
    if (updateInProgress) {
        updateQueued = true;
        return;
    }

    updateInProgress = true;
    // TODO: offset's pre-calculated value is not a good solution, it should be fixed
    let offset = 10.25;
    let point1X = parseFloat(point1.style.left) + offset;
    let point1Y = parseFloat(point1.style.top) + offset;
    let point2X = parseFloat(point2.style.left) + offset;
    let point2Y = parseFloat(point2.style.top) + offset;

    // console.log('Updating points:', point1X, point1Y, point2X, point2Y);

    fetch('/calculate_route', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            point1: { x: point1X, y: point1Y },
            point2: { x: point2X, y: point2Y },
        }),
    })
    .then(response => response.json())
    .then(data => {
        drawCurve(data.path_points);
        document.getElementById('distance-display').textContent = `Distance: ${formatDistance(data.distance)} meters`;
        updateInProgress = false;
        if (updateQueued) {
            updateQueued = false;
            updatePoints();
        }
    })
    .catch(error => {
        console.error('Failed to update points:', error);
        updateInProgress = false;
        if (updateQueued) {
            updateQueued = false;
            updatePoints();
        }
    });
}

function formatDistance(distance) {
    let number = parseFloat(distance);  // Ensure the value is treated as a floating point number
    let formatted = number.toFixed(2);  // Convert the number to a string with two decimal places
    let parts = formatted.split('.');   // Split the string into integer and decimal parts

    // Pad the integer part with spaces if it has fewer than three digits
    while (parts[0].length < 3) {
        parts[0] = ' ' + parts[0];
    }

    return parts[0] + '.' + parts[1];   // Reconstruct the formatted number with spaces and two decimals
}


function drawSmoothCurve(pathPoints) {
    if (pathPoints.length < 2) return;
    lastPathPoints = pathPoints; // Store the latest path points for redraw

    let smoothPathPoints = pathPoints.slice(); // Copy the path points to avoid modifying the original array

    // Add points between the path points to create a smooth curve
    // Add point if distance between two points is greater than 40 pixels
    let i = 0;
    while(i < smoothPathPoints.length - 1) {
        let p1 = smoothPathPoints[i];
        let p2 = smoothPathPoints[i + 1];
        let distance = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]);
        if (distance > 40) {
            let x = (p1[0] + p2[0]) / 2;
            let y = (p1[1] + p2[1]) / 2;
            smoothPathPoints.splice(i + 1, 0, [x, y]);
        }
        else {
            i++;
        }
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.beginPath();
    ctx.moveTo(smoothPathPoints[0][0], smoothPathPoints[0][1]);
    for (let i = 0; i < smoothPathPoints.length - 1; i++) {
        let p0 = smoothPathPoints[i === 0 ? i : i - 1];
        let p1 = smoothPathPoints[i];
        let p2 = smoothPathPoints[i + 1];
        let p3 = smoothPathPoints[i + 2] || p2;

        for (let t = 0; t < 1; t += 0.1) {
            let x = catmullRomInterpolate(p0[0], p1[0], p2[0], p3[0], t);
            let y = catmullRomInterpolate(p0[1], p1[1], p2[1], p3[1], t);
            ctx.lineTo(x, y);
        }
    }

    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 3;
    ctx.stroke();
}

// Function to draw only the points
function drawPoints(pathPoints) {
    lastPathPoints = pathPoints; // Store the latest path points for redraw
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw points
    pathPoints.forEach(point => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'blue';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
    });
}

function drawDiscreteCurve(pathPoints) {
    lastPathPoints = pathPoints; // Store the latest path points for redraw
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.beginPath();
    ctx.moveTo(pathPoints[0][0], pathPoints[0][1]);
    for (let i = 1; i < pathPoints.length; i++) {
        ctx.lineTo(pathPoints[i][0], pathPoints[i][1]);
    }

    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 3;
    ctx.stroke();
}

function catmullRomInterpolate(p0, p1, p2, p3, t) {
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
    );
}

function redrawPath() {
    if (lastPathPoints.length > 0) {
        adjustCanvasSize(); // Make sure the canvas size is updated
        drawCurve(lastPathPoints); // Redraw the curve with the last known path points
    }
}

document.getElementById('smooth-curve-switch').addEventListener('change', function() {
    if (this.checked) {
        drawCurve = drawSmoothCurve;
    } else {
        drawCurve = drawPoints;
    }
    redrawPath();
});

function throttle(func, delay) {
    let lastCall = 0;
    let lastArgs = null;
    let timeout = null;

    return function(...args) {
        const now = (new Date()).getTime();
        lastArgs = args;
        if (now - lastCall < delay) {
            clearTimeout(timeout);
            timeout = setTimeout(() => {
                lastCall = now;
                func.apply(this, lastArgs);
            }, delay);
            return;
        }
        clearTimeout(timeout);
        lastCall = now;
        func.apply(this, args);
    };
}

const throttledUpdatePoints = throttle(updatePoints, 25);
window.addEventListener('resize', adjustCanvasSize);
window.addEventListener('resize', redrawPath);
