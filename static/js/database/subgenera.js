    import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-app.js";
    import { getDatabase, ref, get } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-database.js";
  
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
  
    const app = initializeApp(firebaseConfig);
    const db = getDatabase(app);
  
    // Get speciesKey from the URL
    function getSpeciesKeyFromPath() {
      const pathParts = window.location.pathname.split('/');
      return pathParts[pathParts.length - 1]; // Extract speciesKey from URL
    }
  
    // Fetch species details
    async function fetchDetails() {
      const speciesKey = getSpeciesKeyFromPath();
      if (!speciesKey) {
        console.error('Species key is missing');
        document.getElementById('details-container').innerText = 'Invalid request';
        return;
      }
  
      try {
        // Check in the 'species/subgenera' path for speciesKey
        const snapshot = await get(ref(db, `species/subgenera`));
        if (snapshot.exists()) {
          const subgeneraData = snapshot.val();
          let speciesData = null;
  
          // Search through all subgenera to find the speciesKey
          for (const subgenus in subgeneraData) {
            if (subgeneraData[subgenus][speciesKey]) {
              speciesData = subgeneraData[subgenus][speciesKey];
              break;
            }
          }
  
          if (speciesData) {
            displayDetails(speciesData);
          } else {
            console.warn('Species not found');
            document.getElementById('details-container').innerText = 'Species not found';
          }
        } else {
          console.error('Data not found at the path');
          document.getElementById('details-container').innerText = 'Data not found';
        }
      } catch (error) {
        console.error('Error fetching data:', error);
        document.getElementById('details-container').innerText = 'Error fetching data';
      }
    }
  
    // Display species details in the UI
    function displayDetails(data) {
      const scientificName = document.getElementById('scientificName');
      const scientificAuthorship = document.getElementById('scientificAuthor');
      const taxoRank = document.getElementById('taxonomicRank');
      const taxoStatus = document.getElementById('taxonomicStatus');
      const taxoRemarks = document.getElementById('taxonomicRemarks');
      const reference = document.getElementById('reference');
      const subgenusImage = document.getElementById('subgenus-image');
      const subgenusNameContainer = document.getElementById('subgenus-name'); // The container for subgenus name
  
      // Create a string with the combined name
      const genusName = data.genusName;
      const subGenusName = data.subGenusName || 'Unknown Subgenus';
      const speciesName = data.speciesName || data.speciesKey || 'Unknown Species'; // Use speciesKey if speciesName is not available
  
      // Combine the names with the subgenusName wrapped in a span for styling
      const combinedName = `<span style="font-style: italic; font-weight: 400">${genusName}</span> <span style="font-style: italic; font-weight: 400">${subGenusName}</span> <span style="font-style: italic; font-weight: 400">${speciesName}</span>`;
  
      // Set the inner HTML of the subgenus name container
      subgenusNameContainer.innerHTML = combinedName;
  
      // Apply font size and font family styles to the entire container
      subgenusNameContainer.style.fontSize = "20px";  // Example font size
  
      // Set Scientific Name and Image
      scientificName.innerText = `${genusName} ${subGenusName || 'Unknown Subgenus'} ${speciesName}`;
      scientificAuthorship.innerText = data.scientificAuthorship || 'No authorship available'; // Assuming taxoRemarks is used for this
      subgenusImage.src = data.imageUrl || 'placeholder-image-url.jpg';
  
      // Set Taxonomic Information
      taxoRank.innerText = data.taxoRank || 'Unknown Rank';
      taxoStatus.innerHTML = formatTaxonomicStatus(data.taxoStatus || 'No status available');
      taxoRemarks.innerText = data.taxoRemarks || 'No remarks available';
  
      // Set Reference Information
      const referenceText = data.reference || 'No reference information available'; // Reference field
      reference.innerText = referenceText;  // Set reference as displayed text
  
      // Initialize Map if latitude and longitude data exists
      if (data.lat && data.lon && data.location) {
        initializeMap(data.lat, data.lon, data.location);
      }
    }

    
// Format taxonomic status to a list with italicized values for Genus and Species only
function formatTaxonomicStatus(status) {
  // Split the status into individual lines and format each as a list item
  const formattedItems = status.match(/(Genus:|Species:|Family:|Order:)\s*[^ ]+/g)
    .map(item => {
      const [label, ...value] = item.split(':'); // Separate label and value
      const formattedValue = value.join(':').trim();
      
      // Apply italic style only to Genus and Species
      const valueStyle = (label.trim() === 'Genus' || label.trim() === 'Species') 
                         ? `<span class="taxo-value italic">${formattedValue}</span>`
                         : `<span class="taxo-value">${formattedValue}</span>`;
      
      return `<li><strong>${label}:</strong> ${valueStyle}</li>`;
    }).join('');
  
  // Wrap in a list container
  return `<ul class="taxo-list">${formattedItems}</ul>`;
}
  
    // Initialize Mapbox map with species location
    function initializeMap(latitude, longitude, locationName) {
      mapboxgl.accessToken = 'pk.eyJ1IjoiemVuaXRzdS02IiwiYSI6ImNsejU4Z3Q0ZjQwa2MyanF2dzJ3a2M5YTYifQ.HQjyfHtXRX4Hba4YKlu-qA';
      const map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/streets-v12',
        center: [longitude, latitude],
        zoom: 12 // Set zoom level, can adjust to desired level
      });
  
      // Add navigation controls to the map
map.addControl(new mapboxgl.NavigationControl());

      // Add the marker at the species location
      const marker = new mapboxgl.Marker()
        .setLngLat([longitude, latitude])
        .addTo(map);
  
      // Create and add a popup to the marker
      const popup = new mapboxgl.Popup({ offset: 25 })
      .setHTML(`<strong style="color: black;">Location:</strong> <span style="color: black;">${locationName}</span>`)
      .addTo(map);
  
      // Bind the popup to the marker
      marker.setPopup(popup);
    }
    
  
    // Run the fetch details function on page load
    window.onload = fetchDetails;
  