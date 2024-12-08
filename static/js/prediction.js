document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM fully loaded and parsed');

    // Ensure all elements are correctly selected
    const guideOverlay = document.getElementById('guideOverlay');
    const guideBox = document.getElementById('guideBox');
    const guideNextBtn = document.getElementById('guideNextBtn');
    const guideEndBtn = document.getElementById('guideEndBtn');
    const uploadButton = document.getElementById('imagefile');
    const scanButton = document.querySelector('.btn-primary');
    const predictionResult = document.getElementById('predictionResult');
    const guideIcon = document.getElementById('guideIcon');
    const guideText = document.getElementById('guideText');

    // Log each element to ensure they exist
    console.log({ guideOverlay, guideBox, guideNextBtn, guideEndBtn, uploadButton, scanButton, predictionResult, guideIcon, guideText });

    if (!guideOverlay || !guideBox || !guideNextBtn || !guideEndBtn || !uploadButton || !scanButton || !predictionResult || !guideIcon || !guideText) {
        console.error('One or more elements are missing in the DOM.');
        return;
    }

    // Flag to control guide state
    let step = 0;
    let guideActive = false;

    function startGuide() {
        if (guideActive) return; // Prevent restarting if already active
        guideActive = true;

        // Reset tutorial state
        step = 0;
        guideOverlay.style.display = 'block';
        guideBox.style.display = 'block';
        uploadButton.classList.add('highlight');
        positionGuideBox(uploadButton);
        guideNextBtn.style.display = 'inline-block';
        guideEndBtn.style.display = 'none';
        guideText.innerHTML = "Step 1: Upload an image using the button below.";
    }

    function positionGuideBox(element) {
        const rect = element.getBoundingClientRect();
        guideBox.style.top = `${rect.bottom + 10}px`;
        guideBox.style.left = `${rect.left}px`;
    }

    function nextStep() {
        step++;
        if (step === 1) {
            uploadButton.classList.remove('highlight');
            scanButton.classList.add('highlight');
            positionGuideBox(scanButton);
            guideText.innerHTML = "Step 2: Click the button above to scan the image.";
        } else if (step === 2) {
            scanButton.classList.remove('highlight');
            predictionResult.classList.add('highlight');
            positionGuideBox(predictionResult);
            guideNextBtn.style.display = 'none';
            guideEndBtn.style.display = 'inline-block';
            guideText.innerHTML = "Step 3: View the prediction result.";
        }
    }

    function endGuide() {
        guideActive = false; // Allow restarting the guide
        guideOverlay.style.display = 'none';
        guideBox.style.display = 'none';
        uploadButton.classList.remove('highlight');
        scanButton.classList.remove('highlight');
        predictionResult.classList.remove('highlight');
        step = 0; // Reset step
    }

    // Attach event listeners
    guideIcon.addEventListener('click', startGuide);
    guideNextBtn.addEventListener('click', nextStep);
    guideEndBtn.addEventListener('click', endGuide);

    // Function to display the uploaded image
    function displayImage(event) {
        var image = document.createElement('img');
        image.src = URL.createObjectURL(event.target.files[0]);
        image.alt = "Uploaded Image";

        var imageContainer = document.getElementById('imageContainer');
        imageContainer.innerHTML = ''; // Clear existing content
        imageContainer.appendChild(image);
    }

    // Attach event listener for file input
    uploadButton.addEventListener('change', displayImage);

    // Toggle visibility of the image format section
    document.getElementById('imageFormatBtn').addEventListener('click', function () {
        const formatContainer = document.getElementById('imageFormatOverlay');
        formatContainer.style.display = formatContainer.style.display === 'none' ? 'block' : 'none';
    });

    // Function to show the image format overlay
    function showImageFormat() {
        document.getElementById('imageFormatOverlay').style.display = 'flex';
    }

    // Function to close the image format overlay
    function closeImageFormat() {
        document.getElementById('imageFormatOverlay').style.display = 'none';
    }
});
