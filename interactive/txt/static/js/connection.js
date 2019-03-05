
function sendNoteSequence(noteSequence, genSequenceLen, generationCallback) {
    var formData = new FormData();
    formData.append("noteSequence", noteSequence);
    formData.append("genSequenceLen", genSequenceLen);

    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
            generationCallback(this.responseText);
        }
    }

    request.open('POST', $SCRIPT_ROOT + '/generate', true);
    request.send(formData);
}
