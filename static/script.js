document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("analysisForm");
    if (!form) {
        console.error("Form element with id 'analysisForm' not found.");
        return;
    }
    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        // Get URL input from user form
        const urlInput = event.target.url.value;
        // Config basic UI element (loader, result as text and chart)
        const loader = document.getElementById("loader");
        const resultContainer = document.getElementById("result");
        const sentimentChartCanvas = document.getElementById("sentimentChart");
        // Config dynamic display for loader and result placeholder with state change
        loader.classList.remove("hidden");
        resultContainer.classList.add("hidden");
        // POST URL data to the server and GET json response analysis (message, sentiment, download file)
        try {
            const response = await fetch("/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: `url=${encodeURIComponent(urlInput)}`,
            });
            // Await and prepare json body for the response
            const data = await response.json();
            loader.classList.add("hidden");
            // When response is valid, obtained and connected, prepare sentiment pie-chart and dynamic html body for text
            if (response.ok) {
                const ctx = sentimentChartCanvas.getContext("2d");
                new Chart(ctx, {
                    type: "pie",
                    data: {
                        labels: ["Positive", "Negative"],
                        datasets: [{
                            label: "Sentiment",
                            data: [data.positive, data.negative],
                            backgroundColor: ["#28a745", "#dc3545"],
                        }],
                    },
                });
                // Prepare dynamic html body when obtaining results
                const resultHTML = `
                    <h3>Key Phrases</h3>
                    <p>${data.key_phrases.join(", ")}</p>
                    <h3>Named Entities</h3>
                    <p>${data.named_entities.join(", ")}</p>
                    <h3>Persuasive Contexts</h3>
                    <p>${data.persuasive_contexts.join(", ")}</p>
                    <h3>Summary Contexts</h3>
                    <p>${data.summary_contexts.join(", ")}</p>
                `;
                resultContainer.innerHTML = resultHTML;
                resultContainer.classList.remove("hidden");
            }        
        // Catching request error with message and alert handling
        } catch (error) {
            loader.classList.add("hidden");
            alert(`Error: ${error.message}`);
        }
    });
});

