const textarea = document.getElementById("news_text");

if (textarea) {
    textarea.addEventListener("input", () => {
        if (textarea.value.length > 50000) {
            textarea.value = textarea.value.slice(0, 50000);
        }
    });
}
