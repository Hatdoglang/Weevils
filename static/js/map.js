// Import Firebase SDK functions
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import { getDatabase, ref, get } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-database.js";

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
console.log("Initializing Firebase...");
const app = initializeApp(firebaseConfig);
const db = getDatabase(app);
console.log("Firebase initialized.");


// Mapbox initialization
mapboxgl.accessToken = 'pk.eyJ1IjoiemVuaXRzdS02IiwiYSI6ImNsejU4Z3Q0ZjQwa2MyanF2dzJ3a2M5YTYifQ.HQjyfHtXRX4Hba4YKlu-qA';
const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/streets-v12',
    center: [125.4416, 7.0419],  // Initial map center coordinates
    zoom: 5,
    projection: 'mercator'  // Ensure mercator projection
});

// Add navigation controls to the map
map.addControl(new mapboxgl.NavigationControl());

// Array to store markers for searching
const markers = [];

// Function to get coordinates from a location string using Mapbox Geocoding API
async function getCoordinatesFromLocation(location) {
    const geocodeUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(location)}.json?access_token=${mapboxgl.accessToken}`;
    
    try {
        const response = await fetch(geocodeUrl);
        const data = await response.json();
        
        if (data.features && data.features.length > 0) {
            const coords = data.features[0].center; // [longitude, latitude]
            return { lat: coords[1], lon: coords[0] };
        } else {
            console.error(`No coordinates found for location: ${location}`);
            return null;
        }
    } catch (error) {
        console.error(`Error fetching coordinates for location: ${location}`, error);
        return null;
    }
}

async function loadMarkers() {
    try {
        // Load genus markers
        const genusSnapshot = await get(ref(db, 'species/genera'));
        if (genusSnapshot.exists()) {
            for (const genusKey in genusSnapshot.val()) {
                const genus = genusSnapshot.val()[genusKey];
                console.log("Genus data:", genus);  // Debugging log

                // Iterate over each species in the genus
                for (const speciesKey in genus) {
                    const speciesData = genus[speciesKey];
                    console.log("Species Data:", speciesData); // Debugging log

                    const location = speciesData?.location;  // Access the location for each species
                    const speciesName = speciesData?.speciesName || speciesKey; // Use speciesKey if speciesName is not available
                    const genusLabel = speciesData?.genusName || genusKey; // Use genusKey if genusName is not available
                    const imageUrl = speciesData?.imageUrl; // Image URL if available

                    // Build scientific name: genusKey + speciesKey
                    const scientificName = `${genusKey} ${speciesKey}`;

                    if (location) {
                        const coordinates = await getCoordinatesFromLocation(location);
                        if (coordinates) {
                            console.log(`Coordinates for ${speciesName}:`, coordinates); // Log coordinates
                            // Format popup text with styling (display only scientific name and location)
                            const popupText = `
                                <div style="font-size: 14px; color: #333;">
                                    <strong style="font-size: 16px; color: #0056b3;">Scientific Name:</strong> 
                                    <span style="font-style: italic;">${scientificName}</span><br>
                                    <strong style="color: #006400;">Location:</strong> ${location}
                                </div>
                            `;
                            const marker = createMarker(coordinates.lat, coordinates.lon, popupText, imageUrl);
                            markers.push({ name: speciesName, marker });
                        } else {
                            console.error(`Could not find coordinates for location: ${location}`);
                        }
                    } else {
                        console.error(`Location is missing for species: ${speciesName}`);
                    }
                }
            }
        }

        // Load subgenus markers
        const subgenusSnapshot = await get(ref(db, 'species/subgenera'));
        if (subgenusSnapshot.exists()) {
            for (const subgenusKey in subgenusSnapshot.val()) {
                const subgenus = subgenusSnapshot.val()[subgenusKey];
                console.log("Subgenus data:", subgenus);  // Debugging log

                // Iterate over each species in the subgenus
                for (const speciesKey in subgenus) {
                    const speciesData = subgenus[speciesKey];
                    console.log("Species Data:", speciesData); // Debugging log

                    const location = speciesData?.location;  // Access the location for each species
                    const speciesName = speciesData?.speciesName || speciesKey; // Use speciesKey if speciesName is not available
                    const subgenusLabel = speciesData?.subgenusName || subgenusKey; // Use subgenusKey if subgenusName is not available
                    const genusLabel = speciesData?.genusName || "Unknown Genus"; // Genus name
                    const imageUrl = speciesData?.imageUrl; // Image URL if available

                    // Build scientific name: genusName + subgenusKey + speciesName
                    const scientificName = `${genusLabel} ${subgenusKey} ${speciesName}`;

                    if (location) {
                        const coordinates = await getCoordinatesFromLocation(location);
                        if (coordinates) {
                            console.log(`Coordinates for ${speciesName}:`, coordinates); // Log coordinates
                            // Format popup text with styling (display only scientific name and location)
                            const popupText = `
                                <div style="font-size: 14px; color: #333;">
                                    <strong style="font-size: 16px; color: #0056b3;">Scientific Name:</strong> 
                                    <span style="font-style: italic;">${scientificName}</span><br>
                                    <strong style="color: #006400;">Location:</strong> ${location}
                                </div>
                            `;
                            const marker = createMarker(coordinates.lat, coordinates.lon, popupText, imageUrl);
                            markers.push({ name: speciesName, marker });
                        } else {
                            console.error(`Could not find coordinates for location: ${location}`);
                        }
                    } else {
                        console.error(`Location is missing for subgenus species: ${speciesName}`);
                    }
                }
            }
        }
    } catch (error) {
        console.error("Error loading markers:", error);
    }
}


// Adjust map view to fit all markers
function fitMapToMarkers() {
    if (markers.length > 0) {
        const bounds = new mapboxgl.LngLatBounds();
        markers.forEach(markerData => {
            bounds.extend(markerData.marker.getLngLat());
        });
        map.fitBounds(bounds, { padding: 50 });
    }
}

// Function to create a marker with an offset for overlapping markers
function createMarker(lat, lon, popupText, imageUrl) {
    const existingMarker = markers.find(markerData => 
        markerData.marker.getLngLat().lat === lat && markerData.marker.getLngLat().lng === lon
    );

    // Apply offset if a marker already exists at this location
    const offset = existingMarker ? 0.0001 : 0; // If marker exists, apply a slight offset to the position

    const marker = new mapboxgl.Marker()
        .setLngLat([lon + offset, lat + offset]) // Apply offset to longitude and latitude
        .addTo(map);

    // Add popup to marker
    const popup = new mapboxgl.Popup({ offset: 25 })
        .setHTML(`<h3>${popupText}</h3><img src="${imageUrl}" alt="Species Image" style="width:100px; height:auto;" />`);
    marker.setPopup(popup);

    // Store marker data
    markers.push({ marker: marker, lat: lat + offset, lon: lon + offset });

    return marker;
}

// Call fitMapToMarkers after loading markers
map.on('load', function () {
    loadMarkers().then(() => {
        fitMapToMarkers();
    });
});



let isUserInteracted = false;

// Array to store weevil names and markers for species
let allWeevilNames = [];
let speciesMarkers = {};

// Function to load weevil names from the database and populate allWeevilNames and speciesMarkers
async function loadWeevilNames() {
    try {
        const genusSnapshot = await get(ref(db, 'species/genera'));
        if (genusSnapshot.exists()) {
            console.log("Genus Data Loaded:", genusSnapshot.val());
            for (const genusKey in genusSnapshot.val()) {
                const genus = genusSnapshot.val()[genusKey];
                for (const speciesKey in genus) {
                    const speciesData = genus[speciesKey];
                    const location = speciesData?.location;
                    const speciesName = speciesData?.speciesName;  // Assuming speciesName is available
                    if (location && speciesName) {
                        const name = `${genusKey} ${speciesName}`;  // Format name for search
                        allWeevilNames.push({ name, format: 'genus', genusName: genusKey, speciesName });
                        speciesMarkers[name] = { genusName: genusKey, speciesName, location };
                    }
                }
            }
        }

        const subgenusSnapshot = await get(ref(db, 'species/subgenera'));
        if (subgenusSnapshot.exists()) {
            console.log("Subgenus Data Loaded:", subgenusSnapshot.val());
            for (const subgenusKey in subgenusSnapshot.val()) {
                const subgenus = subgenusSnapshot.val()[subgenusKey];
                for (const speciesKey in subgenus) {
                    const speciesData = subgenus[speciesKey];
                    const location = speciesData?.location;
                    const speciesName = speciesData?.speciesName;  // Assuming speciesName is available
                    const genusName = speciesData?.genusName; // Assuming genusName is also available
                    if (location && speciesName && genusName) {
                        const name = `${genusName} ${subgenusKey} ${speciesName}`;  // Format full name
                        allWeevilNames.push({ name, format: 'subgenus', genusName, subgenusName: subgenusKey, speciesName });
                        speciesMarkers[name] = { genusName, subgenusName: subgenusKey, speciesName, location };
                    }
                }
            }
        }
        console.log("All Weevil Names Loaded:", allWeevilNames);
    } catch (error) {
        console.error("Error loading weevil names:", error);
    }
}

// Function to handle input and show suggestions
document.getElementById('weevil-search').addEventListener('input', function(event) {
    const query = event.target.value.toLowerCase().trim();
    const suggestionsList = document.getElementById('suggestions-list');
    suggestionsList.innerHTML = ''; // Clear previous suggestions

    // Sanitize query to handle special characters like parentheses
    const sanitizedQuery = query.replace(/[()]/g, ""); // Remove parentheses for search comparison
    
    // Filter the weevil names based on partial matching with the sanitized query
    const filteredNames = allWeevilNames.filter(item => item.name.toLowerCase().includes(sanitizedQuery));
    
    if (query.length > 0 && filteredNames.length > 0) {
        filteredNames.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item.name;
            li.addEventListener('click', () => {
                document.getElementById('weevil-search').value = item.name;
                suggestionsList.style.display = 'none';
                zoomToSpeciesLocation(item.name);
                isUserInteracted = true; // Set flag to true after user interacts
            });
            suggestionsList.appendChild(li);
        });
        suggestionsList.style.display = 'block'; // Show the suggestions
    } else {
        suggestionsList.style.display = 'none'; // Hide suggestions if no match
    }
});

// Function to zoom map to the selected species location
function zoomToSpeciesLocation(speciesName) {
    const speciesData = speciesMarkers[speciesName];
    if (speciesData) {
        getCoordinatesFromLocation(speciesData.location).then(coordinates => {
            if (coordinates && isUserInteracted) { // Only zoom if user has interacted
                zoomMapToCoordinates(coordinates.lat, coordinates.lon);
            } else {
                console.error(`Coordinates not found for location: "${speciesData.location}"`);
            }
        });
    } else {
        console.warn(`Species data not found for "${speciesName}".`);
    }
}

// Function to zoom smoothly to coordinates using Mapbox's flyTo
function zoomMapToCoordinates(lat, lon) {
    if (!window.map) {
        console.error("Map instance is not initialized.");
        return;
    }
    window.map.flyTo({
        center: [lon, lat],
        zoom: 20,  // Adjust zoom level if needed
        speed: 1.2,
        curve: 1.5,
        essential: true  // Ensures the animation is not interrupted
    });
}

// Make map globally accessible
window.map = map; // Ensure the map instance is available globally

// Load weevil names and markers on page load
loadWeevilNames().then(() => {
    console.log("Weevil data loaded successfully.");
});
