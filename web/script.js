function selectFile() {
    eel.select_file();
}


function switchTheme() {
    var themeContainer = document.getElementById('theme-body');
    if (themeContainer.classList.contains('light-theme')) {
        themeContainer.classList.remove('light-theme');
        themeContainer.classList.add('dark-theme');
    } else {
        themeContainer.classList.remove('dark-theme');
        themeContainer.classList.add('light-theme');
    }
    var themeIcon = document.getElementById('theme-icon');
    if (themeContainer.classList.contains('light-theme')) {
        themeIcon.src = 'assets/media-luna.png'; // Ruta de la imagen para el tema claro
    } else {
        themeIcon.src = 'assets/sun.png'; // Ruta de la imagen para el tema oscuro
    }
}

function showPrediction(className, probability) {
    if (probability !== null && probability !== undefined) {
        probability = parseFloat(probability);
        document.getElementById('prediction').innerHTML = `Predicción: ${className}, Probabilidad: ${probability.toFixed(2)}`;
    } else {
        console.error('La probabilidad es nula o no definida.');
    }
}

function showImage(imageData) {
    document.getElementById('image').src = `data:image/png;base64,${imageData}`;
    document.getElementById('container').style.display = 'block';
}

eel.expose(showImage);
eel.expose(showPrediction);
