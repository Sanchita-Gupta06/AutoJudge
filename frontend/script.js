async function predict() {
    const btn = document.getElementById("analyzeBtn");
    btn.disabled = true;
    btn.textContent = "Analyzing...";

    const payload = {
        description: desc.value,
        input_description: input.value,
        output_description: output.value
    };

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await res.json();

    btn.disabled = false;
    btn.textContent = "Analyze Difficulty";

    if (data.error) {
        alert(data.error);
        return;
    }

    document.getElementById("result").classList.remove("hidden");

    // Badge
    const badge = document.getElementById("difficultyBadge");
    badge.className = `badge ${data.predicted_class}`;
    badge.textContent = data.predicted_class.toUpperCase();

    // Score
    document.getElementById("score").textContent = data.predicted_score;

    // Confidence
    const confidencePercent = Math.round(data.confidence * 100);
    document.getElementById("confidenceText").textContent = confidencePercent + "%";
    document.getElementById("confidenceFill").style.width = confidencePercent + "%";

    // Highlights
    const hl = document.getElementById("highlights");
    hl.innerHTML = "";
    if (data.highlights.length === 1 && data.highlights[0].includes("straightforward")) {
        hl.innerHTML = `<i>${data.highlights[0]}</i>`;
    } else {
        data.highlights.forEach(word => {
            hl.innerHTML += `<span>${word}</span>`;
        });
    }

    // Similar problems
    const sim = document.getElementById("similar");
    sim.innerHTML = "";
    if (!data.similar_problems || data.similar_problems.length === 0) {
        sim.innerHTML = "<i>No similar problems found.</i>";
    } else {
        data.similar_problems.forEach(p => {
            sim.innerHTML += `<div>
                <strong>${p.title}</strong><br/>
                Difficulty: ${p.class} • Score: ${p.score}
            </div>`;
        });
    }
}
