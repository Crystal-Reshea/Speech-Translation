from flask import Flask, render_template
from flask import request
from asr_inference import ASRInference
import soundfile as sf 

app = Flask(__name__, template_folder='templates')
asr = ASRInference()

@app.route("/", methods=["POST", "GET"])
def index():
    text = ""

    if request.method=="POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename=="":
            return redirect(request.url)
        
        if file:
            audio, fs = sf.read(file)
            text = asr.inference(audio)
            source = request.form.get('srcLang')
            target = request.form.get('targetLang')
            translated_text = asr.get_languages(str(source), str(target))
    return render_template("upload.html", text = translated_text)


if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
#     target = 
#             translated_text = asr.get_languages("english","spanish")