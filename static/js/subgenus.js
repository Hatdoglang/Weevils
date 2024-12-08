import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-app.js";
import { getDatabase, ref, get } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-database.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-analytics.js";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDyurkD5mkFM_TAi3TmLOE3boehVNtNJFY",
  authDomain: "capstone-92833.firebaseapp.com",
  databaseURL: "https://capstone-92833-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "capstone-92833",
  storageBucket: "capstone-92833.appspot.com",
  messagingSenderId: "824130896942",
  appId: "1:824130896942:web:f2cc74327abd0e9c304bbf",
  measurementId: "G-L57NFDXD3B"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const db = getDatabase(app);

let speciesData = []; // Store fetched data globally

// Function to fetch and display data
async function fetchData() {
  try {
    const snapshot = await get(ref(db, 'species/subgenera')); // Fetch subgenera data
    if (snapshot.exists()) {
      const subgeneraData = snapshot.val();
      speciesData = []; // Reset species data array

      // Process subgenera data
      for (const subgenusKey in subgeneraData) {
        const subgenus = subgeneraData[subgenusKey];
        for (const speciesKey in subgenus) {
          const species = subgenus[speciesKey];

          // Construct full name, using speciesKey if speciesName is not available
          const speciesName = species.speciesName ? species.speciesName : speciesKey;
          const fullName = `${species.genusName} ${subgenusKey} ${speciesName}`;

          speciesData.push({
            name: fullName,
            imageUrl: species.imageUrl,
            key: speciesKey
          });
        }
      }
      displayData(speciesData); // Display fetched data
    } else {
      console.log('No data available');
    }
  } catch (error) {
    console.error('Error fetching data:', error);
  }
}


// Function to display data in the gallery
function displayData(data) {
  const galleryContainer = document.getElementById('gallery');
  galleryContainer.innerHTML = ''; // Clear any existing content

  data.forEach(item => {
    // Create HTML elements for each item
    const colDiv = document.createElement('div');
    colDiv.className = 'col-xl-3 col-lg-4 col-md-6';

    const galleryItem = document.createElement('div');
    galleryItem.className = 'gallery-item h-100 position-relative';

    const img = document.createElement('img');
    img.src = item.imageUrl;
    img.alt = item.name;
    img.className = 'img-fluid';

    const linksDiv = document.createElement('div');
    linksDiv.className = 'gallery-links d-flex align-items-center justify-content-center';

    const previewLink = document.createElement('a');
    previewLink.href = item.imageUrl;
    previewLink.title = item.name;
    previewLink.className = 'glightbox preview-link';
    previewLink.innerHTML = '<i class="bi bi-arrows-angle-expand"></i>';
    linksDiv.appendChild(previewLink);

    // Assuming `item` represents a species object from Firebase
    const speciesKey = item.key; // Ensure this is the unique species key within the subgenus

    // Construct the link to the subgenus details page
    const detailsLink = document.createElement('a');
    detailsLink.href = `/subgenusdetails/${encodeURIComponent(speciesKey)}`; // Correctly encoding the species key
    detailsLink.className = 'details-link';
    detailsLink.innerHTML = '<i class="bi bi-link-45deg"></i>';
    linksDiv.appendChild(detailsLink);


    const nameDiv = document.createElement('div');
    nameDiv.className = 'gallery-name';
    nameDiv.innerText = item.name;

    // Append all elements
    galleryItem.appendChild(img);
    galleryItem.appendChild(linksDiv);
    galleryItem.appendChild(nameDiv);

    colDiv.appendChild(galleryItem);
    galleryContainer.appendChild(colDiv);
  });
}

// Function to handle input and show suggestions
const suggestionsList = document.getElementById('suggestions-list');

function searchImage() {
  const searchInput = document.getElementById('imageSearch').value.toLowerCase();
  const filteredData = speciesData.filter(item => item.name.toLowerCase().includes(searchInput));

  displayData(filteredData); // Display the filtered results

  const notification = document.getElementById('notification'); // Get the notification element
  if (filteredData.length === 0) {
    notification.innerHTML = `Sorry! No Results Found for <span class="red-text">(${searchInput})</span><br>Please use exact or shortest keyword...<br>to get better search results`;
    notification.style.display = 'block'; // Show notification
  } else {
    notification.style.display = 'none'; // Hide notification if results found
  }

  // Show suggestions based on input
  showSuggestions(searchInput);
}

// Function to show suggestions based on the input
function showSuggestions(input) {
  suggestionsList.innerHTML = ''; // Clear previous suggestions

  if (input.length > 0) {
    const filteredNames = speciesData.filter(item => item.name.toLowerCase().includes(input)); // Filter based on the input

    // Display suggestions if there are matches
    if (filteredNames.length > 0) {
      filteredNames.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item.name;
        li.addEventListener('click', () => {
          document.getElementById('imageSearch').value = item.name; // Set input value to selected suggestion
          suggestionsList.style.display = 'none'; // Hide suggestions after selection
          searchImage(); // Trigger the search based on the selected name
        });
        suggestionsList.appendChild(li);
      });
      suggestionsList.style.display = 'block'; // Show suggestions
    } else {
      suggestionsList.style.display = 'none'; // Hide suggestions if no matches
    }
  } else {
    suggestionsList.style.display = 'none'; // Hide suggestions if input is empty
  }
}

// Event listener for Enter key press to trigger the search when the user presses Enter
document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('imageSearch');
  searchInput.addEventListener('keypress', function (event) {
    if (event.key === 'Enter') {
      searchImage(); // Trigger search on Enter key press
    }
  });
});

// Event listener for input changes to show suggestions
document.getElementById('imageSearch').addEventListener('input', function () {
  const query = this.value.toLowerCase();
  showSuggestions(query); // Show suggestions as the user types
});


// Fetch data when the window loads
window.onload = fetchData;
