class MyModel {

    constructor() {
        this.modelUrl = "./model/model.json";
        this.modelVersion = "4.2.1";
        tf.setBackend("webgl");
        console.log("Tensorflow backend: " + tf.getBackend());

    }

    async loadModel() {
        this.model = await tf.loadGraphModel(this.modelUrl);
    }

    /**
     * Takes an ImageData object and reshapes it to fit the model
     * @param {tf.tensor} pixelData
     */
    async preprocessImage(pixelData) {
        const targetDim = 224;

        let tensor = tf.image.resizeBilinear(pixelData, [targetDim, targetDim]);
        tensor = tensor.expandDims(0); // Reshape again to fit training model [N:=1, 224, 224, 3]

        return tensor;
    }

    /**
     * Takes an ImageData objects and predict a character
     * @param {ImageData} pixelData
     * @param {number} threshold
     * @returns {string | void} gender
     */
    async getPredict(pixelData, threshold = 0.649) {

        const preprocessedInput = await this.preprocessImage(pixelData);

        const result = this.model.predict(preprocessedInput);
        const gender = result.data().then(res => {
                return res > threshold ? "male" : "female";
            }
        );

        return gender;

    }
}

class PredictionApp {

    constructor(fileId = "file-input",
                imgId = "fileImage",
                serverMsgId = "serverMsg") {

        this.imgId = imgId;
        this.serverMsgId = serverMsgId;
        this.fileId = fileId;
        this.resultElement = document.getElementById(this.serverMsgId);
        this.model = new MyModel();

    }

    async onClickWarmUp(event = null) {
        if (!this.model.model) {
            this.model.loadModel().then(() => {
                const warmupResult = this.model.model.predict(tf.zeros([1, 224, 224, 3]));
                warmupResult.data().then(res =>
                    console.log(
                        "Warmed up!!! Result for zeros: " + res));
                warmupResult.dispose();
            });

        }
    }

    async previewFile(event = null) {

        const preview = document.getElementById(this.imgId);
        const files = document.getElementById(this.fileId);

        if (files) {

            const file = files.files[0];

            const reader = new FileReader();
            reader.onload = () => {
                preview.src = reader.result.toString(); // show image in <img> tag

            };

            document.getElementById(this.serverMsgId).innerHTML = "";
            reader.readAsDataURL(file);
        }
    }

    async processImage(event = null) {

        this.modelVersionToDisplay = this.model.modelVersion;
        this.resultElement.innerHTML = "Processing Image ...";

        const image = document.getElementById(this.imgId);
        const tfImg = tf.browser.fromPixels(image);
        const result = await this.model.getPredict(tfImg);
        tfImg.dispose();
        this.showPrediction(result);
    }

    showPrediction(prediction) {
        document.getElementById("model_version").innerHTML = this.modelVersionToDisplay;
        document.getElementById(this.serverMsgId).innerHTML = "This is a picture of a " + prediction;
    }
}

let myApp = new PredictionApp;
