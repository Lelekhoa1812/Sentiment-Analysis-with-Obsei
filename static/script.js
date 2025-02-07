document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("analysisForm");
    if (!form) {
        console.error("Form element with id 'analysisForm' not found.");
        return;
    }
    let sentimentChart; // Store the chart instance globally
    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        // Get URL input from user form
        const urlInput = event.target.url.value;
        // Config basic UI element (loader, result as text and chart)
        const loader = document.getElementById("loader");
        const resultContainer = document.getElementById("result");
        const sentimentChartCanvas = document.getElementById("sentimentChart");
        sentimentChartCanvas.style.width = "400px";
        sentimentChartCanvas.style.height = "400px";
        const resultTextContainer = document.getElementById("resultText"); 
        // Print out any possible errors with module loading (e.g., chart, text container when not loaded correctly)
        if (!sentimentChartCanvas || !resultTextContainer) {
            console.error("Required elements not found", { sentimentChartCanvas, resultTextContainer });
            return;
        }
        if (typeof Chart === "undefined") {
            console.error("Chart.js not loaded.");
        } else {
            console.log("Chart.js loaded.");
        }
        // Config dynamic display for loader and result placeholder with state change
        loader.classList.remove("hidden");
        resultContainer.classList.add("hidden");
        resultTextContainer.innerHTML = ""; // Clear previous results
        // POST URL data to the server and GET json response analysis (message, sentiment, download file)
        try {
            const response = await fetch("/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: `url=${encodeURIComponent(urlInput)}`,
            });
            if (sentimentChart) { // Double checking removal of chart
                console.error("Previous sentiment chart is found and so will be destroyed.");
                sentimentChart.destroy();
            }
            // Await and prepare json body for the response
            const data = await response.json();
            loader.classList.add("hidden");
            // When response is valid, obtained and connected, prepare sentiment pie-chart and dynamic html body for text
            if (response.ok) {
                // Destroy existing chart if it exists
                if (sentimentChart) {
                    console.error("Previous sentiment chart is found and so will be destroyed.");
                    sentimentChart.destroy();
                }
                // Initialize sentiment analysis pie chart with canvas
                const ctx = sentimentChartCanvas?.getContext("2d");
                sentimentChartCanvas.width = sentimentChartCanvas.width; // Reset canvas dimensions to clear it
                // Catching error for invalid context or issue with chart rendering
                if (!ctx) {
                    console.error("Canvas context not found.");
                    return;
                }     
                try {           
                    new Chart(ctx, {
                        type: "pie",
                        data: {
                            labels: ["Positive", "Negative", "Neutral"],
                            datasets: [{
                                label: "Sentiment",
                                data: [data.positive, data.negative, data.neutral],
                                backgroundColor: ["#009999", "#dc3545", "#e5ce20"],
                            }],
                        },
                        // Add tooltip reflecting message to sentiment state
                        options: {
                            plugins: {
                                tooltip: {
                                    callbacks: {
                                        label: function (context) {
                                            const sentiment = context.label;
                                            const value = context.raw.toFixed(2);
                                            let description = "";
                                            if (sentiment === "Positive") {
                                                description = "Positive sentiment indicating a favorable context.";
                                            } else if (sentiment === "Negative") {
                                                description = "Negative sentiment suggesting an unfavorable context.";
                                            } else if (sentiment === "Neutral") {
                                                description = "Neutral sentiment indicating a balanced or mixed context.";
                                            }
                                            return `${sentiment}: ${value}% - ${description}`;
                                        },
                                    },
                                },
                            },
                        },
                    });
                } catch (chartError) {
                    console.error("Error initializing the chart:", chartError);
                    alert("Failed to render the chart. Please try again.");
                }
                console.log("Chart.js Data:", { positive: data.positive, negative: data.negative, neutral: data.neutral });
                console.log("Sentiment Chart Canvas:", sentimentChartCanvas);
                // Dynamic html tag for extreme negative sentence with content and score
                const extremeNegativesHTML = data.extreme_negative_sentences
                    .map(item => `${item.sentence} (${(item.score * 100).toFixed(2)}%) (Fact check: ${(item.is_true)}).`)
                    .join("<br>");
                // Initialize the html body for persuasive contexts div
                let persuasiveContextsHTML = ""; // Clear previously existed data
                // Dynamic html tag for persuasive contexts
                if (data.persuasive_contexts && data.persuasive_contexts.length > 0) {
                    persuasiveContextsHTML = `
                        <h3>Persuasive Contexts</h3>
                        <p>${data.persuasive_contexts.join("<br>")}</p>
                    `;
                }
                // Prepare dynamic html body when obtaining results
                const resultHTML = `
                    <h3>Article Language</h3>
                    <p>${data.language}</p>
                    <h3>Key Phrases</h3>
                    <p>${data.key_phrases.join(", ")}</p>
                    <h3>Named Entities</h3>
                    <p>${data.named_entities.join("<br>")}</p>
                    ${persuasiveContextsHTML}
                    <h3>Extreme Negative Contents</h3>
                    <p>${extremeNegativesHTML}</p>
                    <h3>Summary Contexts</h3>
                    <p>${data.summary_contexts}</p>
                `;
                // Catching error if text result not found
                if (!resultTextContainer) {
                    console.error("Result text container not found.");
                    return;
                }
                resultTextContainer.innerHTML = resultHTML;
                resultContainer.classList.remove("hidden");
            }
            // Catching error on response request
            else {
                alert(`Response Error: ${response}`);
            }       
        // Catching request error with message and alert handling
        } catch (error) {
            loader.classList.add("hidden");
            alert(`Error: ${error.message}`);
        }
    });
});

