
function sendNoteSequence(txtSequence, genSequenceLen, generationCallback) {
    var formData = new FormData();
    formData.append("txtSequence", txtSequence);
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
