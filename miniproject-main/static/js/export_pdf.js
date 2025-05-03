document.getElementById("exportPDF").addEventListener("click", () => {
    const { jsPDF } = window.jspdf;
    let doc = new jsPDF();

    // Set Title
    doc.setFontSize(18);
    doc.text("News Analysis Report", 10, 10);

    // Add Scraped Content
    let articleText = document.getElementById("articleContent").innerText;
    doc.setFontSize(12);
    doc.text("Scraped Content:", 10, 20);
    doc.setFontSize(10);
    doc.text(articleText, 10, 30, { maxWidth: 180 });

    // Add Bag of Words Matrix
    let bowText = document.getElementById("topBowWords").innerText;
    doc.setFontSize(12);
    doc.text("Top 5 Words by Count:", 10, 70);
    doc.setFontSize(10);
    doc.text(bowText, 10, 80, { maxWidth: 180 });

    // Add TF-IDF Matrix
    let tfidfText = document.getElementById("topTfidfWords").innerText;
    doc.setFontSize(12);
    doc.text("Top 5 Words by TF-IDF Value:", 10, 110);
    doc.setFontSize(10);
    doc.text(tfidfText, 10, 120, { maxWidth: 180 });

    // Add Sentiment Analysis
    let sentiment = document.getElementById("sentimentResult").innerText;
    doc.setFontSize(12);
    doc.text("Sentiment Analysis:", 10, 150);
    doc.setFontSize(10);
    doc.text(sentiment, 10, 160, { maxWidth: 180 });

    // Convert Ridge Regression Graph to Image
    let imgElement = document.getElementById("ridgeGraph");
    let imgData = imgElement.src;
    
    let img = new Image();
    img.src = imgData;
    img.onload = function () {
        doc.addImage(img, 'PNG', 15, 180, 80, 60);
        doc.save("news_analysis_report.pdf");
    };
});
