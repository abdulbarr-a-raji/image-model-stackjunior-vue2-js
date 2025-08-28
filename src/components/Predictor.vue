<template>
  <div class="training">
    <h1>{{ header }}</h1>
    <button id="inference-button" v-on:click="run">Run Predictor</button>
    <br>
    <button id="loaded-model-button" v-on:click="this.loadAndPredict">
      Predict with faulty loaded model
    </button>

    <div class="image-container">
        <div id="example1" title="Example #1" class="example-pics"></div>
        <div id="example2" title="Example #2" class="example-pics"></div>
        <div id="example3" title="Example #3" class="example-pics"></div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Predictor',
  props: {
    header: String,
    w: {type:Number, required: true},
    h: {type:Number, required: true}
  },
  mounted() {
    this.tf = window.tf;
  },
  data() {
    return {
      faultyModel: null
    }
  },
  methods: {
    // Checks if tf.js is available in the component, as the name suggests
    checkForTFJS: function () {
      if (this.tf) {
        console.log("tf.js IS in this scope :D");
      } else {
        console.error("tf.js is NOT in this scope D':");
      }
    },
    // given a div element's id, it extracts a tensor from the div's background image
    getExample: function (id) {
      const divElement = document.getElementById(id);
      const divStyles = getComputedStyle(divElement);
      const imageUrl = divStyles.backgroundImage;
      const intWidth = parseInt(divStyles.width.replace("px", ""));
      const intHeight = parseInt(divStyles.height.replace("px", ""));

      const img = new Image();
      img.src = imageUrl.slice(5, -2); // .slice() removes the url("") wrapper text;

      const canvas = document.createElement('canvas');
      canvas.width = intWidth;

      canvas.height = intHeight;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, intWidth, intHeight);

      const imageTensor = this.tf.browser.fromPixels(canvas)
          .resizeBilinear([this.w, this.h])
          .toFloat()
          .div(255.0);

      return imageTensor;
    },
    loadModel: async function () {
      const model = await this.tf.loadLayersModel('/assets/pre-trained model/pretrained-model-v1.json'); // new ERROR with loading caused by movement of files

      console.log("Model loaded:", model);
      console.log("YAY loaded")

      return model;
    },
    makeInferences: function (trainedModel, imgTensors) {
      const output = trainedModel.predictOnBatch(imgTensors);
      return output;
    },
    decodePredictions: async function (preds, legend = {"sad":0, "smiling":1}) {
      const decoded = [];
      const vals = await preds.argMax(-1).array(); // categorises model predictions
      for(let val of vals) {
        console.log(val);
        for(let [classification, code] of Object.entries(legend)) {
          if (val === code) {
            decoded.push(classification);
            break;
          }
        }
      }

      return decoded;
    },
    async loadAndPredict() {
      this.faultyModel = await this.tf.loadLayersModel('/assets/faulty saved model/faulty_model.json');
      // console.log(this.faultyModel.layers);
      this.faultyModel.summary();

      // loading feature extractor
      const MobileNetV2URL =
        "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";

      // Load the MobileNet model using Google's optimized URL (like the fast example)
      const directlyLoaded = await this.tf.loadLayersModel(MobileNetV2URL);

      // Create feature extraction model (like the fast example)
      const beforeFinalLayer = directlyLoaded.getLayer(
        "global_average_pooling2d_1"
      );

      const loadedModel = this.tf.model({
        inputs: directlyLoaded.inputs,
        outputs: beforeFinalLayer.output,
      });

      console.log("MobileNet base feature extractor created");

      // Warm up the model with zeros (memory optimization)
      this.tf.tidy(() => {
        const warmupInput = this.tf.zeros([1, 224, 224, 3]);
        const answer = loadedModel.predict(warmupInput);
        console.log("Warmup completed, feature shape:", answer.shape);
      });

      const tensors_batch = [];
      for(let ex of ['example1', 'example2', 'example3']) {
        tensors_batch.push(loadedModel.predict(this.getExample(ex).expandDims()).squeeze());
      }
      console.log(tensors_batch);
      const batch_tensor = this.tf.stack(tensors_batch);

      const inferences = this.makeInferences(this.faultyModel, batch_tensor);
      inferences.print();
      console.log("decoded:", await this.decodePredictions(inferences));
    },
    run: async function () {
      const tensors_batch = [];
      for(let ex of ['example1', 'example2', 'example3']) {
        tensors_batch.push(this.getExample(ex));
      }
      const batch_tensor = this.tf.stack(tensors_batch);

      const predictor = await this.loadModel();
      const inferences = this.makeInferences(predictor, batch_tensor);

      inferences.print();
      console.log("decoded:", await this.decodePredictions(inferences));
    }
  }
}
</script>

<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
div.image-container {
    width: 40%;
    height: auto;
    padding: 50px;
    background-color: #e9e9e9;
    background-size: 100% 100%;
    /* position: relative;
    top: 300px; */
}

div.example-pics {
    width: 300px;
    height: 300px;
    margin: 50px auto;
    background-repeat: no-repeat;
    background-size: 100% 100%;
}
#example1 {
  background-image: url('/assets/sad-old-female-genai.jpg');
}
#example2 {
  background-image: url('/assets/smiling-young-female-genai.jpg');
}
#example3 {
  background-image: url('/assets/smiling-young-male-genai.jpeg');
}
</style>
