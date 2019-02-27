
function sendNoteSequence(noteSequence) {
    var formData = new FormData();
    formData.append("noteSequence", noteSequence);

    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
            console.log(this.responseText);
        }
    }

    request.open('POST', $SCRIPT_ROOT + '/generate');
    request.send(formData);
}
