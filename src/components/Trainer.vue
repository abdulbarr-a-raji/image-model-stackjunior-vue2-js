<template>
  <div class="training">
    <h1>{{ header }}</h1>
    <h4>CNN Model Demo</h4>
    <div id="micro-out-div">uses tf.js for ML</div>
    <br>
    <input type="file" id="imagedataUpload" multiple accept="image/*" />
    <button id="load-images" v-on:click="runMain">Load Image Data & Train Model</button>
    <hr>
    <button id="train-with-extractor" v-on:click="fitClassifcationHead">
      Load Image Data & Train Classification Head
    </button>
  </div>
</template>

<script>
export default {
  name: 'Training',
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
      tf: null,
      featureExtractor: null,
      fc_head: null,

      // FLAGS
      trainingWithFeatureExtractor: false
    }
  },
  methods: {
    loadImages: async function () {
      const imageTensors = [];
      const labels = [];
      
      const inputElement = document.getElementById('imagedataUpload');
      const files = Array.from(inputElement.files);

      for (const file of files) {
        const label = file.name.includes('smiling') ? 1 : 0;

        const img = await this.readImageFileAsImage(file);
        const tensor = this.preprocessImage(img);
        
        imageTensors.push(tensor);
        labels.push(label);
      }

      /* must catch this error:
        Uncaught (in promise) Error: 
        Pass at least one tensor to tf.stack
        at loadImages (data.js)

        | it occurs when no images are uploaded, 
        | but button has been pressed
      */
      // console.log(imageTensors);
      console.log(imageTensors.length);

      const xs = this.tf.stack(imageTensors);  // tensor shape: [batch, height, width, channels]
      const ys = this.tf.tensor1d(labels, 'int32'); // tensor shape: [batch]
      document.getElementById('micro-out-div').innerText = labels;

      return [xs, ys];
    },
    readImageFileAsImage: function (file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = () => {
          const img = new Image();
          img.src = reader.result;

          img.onload = () => resolve(img);
          img.onerror = reject;
        };

        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    },
    checkFortf: function () {
      if (this.tf) {
        console.log("tf.js is in this scope :D");
      } else {
        console.error("tf.js is NOT in this scope D':");
      }
    },
    preprocessImage: function (img) {
      switch(this.trainingWithFeatureExtractor) {
        case true:
          return this.tf.tidy(() => {
            const tns = this.tf.browser.fromPixels(img);
            
            // setting crop size operation...
            const widthToHeight = tns.shape[1] / tns.shape[0];
            let croppedSize;

            if (widthToHeight > 1) {
              // Image is wider than tall? crop sides
              const heightToWidth = tns.shape[0] / tns.shape[1];
              const cropTop = (1 - heightToWidth) / 2;
              const cropBottom = 1 - cropTop;
              croppedSize = [[cropTop, 0, cropBottom, 1]];
            } else {
              // Image is taller than wide? crop top & bottom
              const cropLeft = (1 - widthToHeight) / 2;
              const cropRight = 1 - cropLeft;
              croppedSize = [[0, cropLeft, 1, cropRight]];
            }

            // crop, resize, and more...
            const croppedImgTns = this.tf.image.cropAndResize(
              tns.expandDims(0),
              croppedSize,
              [0],
              [224, 224]
            ).toFloat()
              .div(255.0);
            
            // disposal/s
            tns.dispose();

            return this.featureExtractor.predict(croppedImgTns).squeeze();
          });
        case false:
          return this.tf.tidy(() => {
            return this.tf.browser.fromPixels(img)
            .resizeBilinear([this.w, this.h])
            .toFloat()
            .div(this.tf.scalar(255));
          });
        default:
          return null;
      }
    },
    getCNNModel: function () {            // getCNNModel function
      const model = this.tf.sequential();

      const IMAGE_DEPTH = 3; // ie. RGB

      const optimizer = this.tf.train.adam();

      model.add(this.tf.layers.conv2d({
        inputShape: [this.w, this.h, IMAGE_DEPTH],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }));

      model.add(this.tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
      
      model.add(this.tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }));
      model.add(this.tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

      model.add(this.tf.layers.flatten());

      model.add(this.tf.layers.dense({
        units: 1,
        kernelInitializer: 'varianceScaling',
        activation: 'sigmoid'
      }));

      model.compile({
        optimizer: optimizer,
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
      });

      return model;

    },
    async fitClassifcationHead() {
      this.trainingWithFeatureExtractor = true;
      const NUM_CLASSES = ['sad', 'smiling'].length;

      // load feature extractor
      try {
        const extractorURL =
          "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";
        const loadedModel = await this.tf.loadLayersModel(extractorURL);

        // Create feature extraction model
        const beforeFinalLayer = loadedModel.getLayer(
          "global_average_pooling2d_1"
        );
        this.featureExtractor = this.tf.model({
          inputs: loadedModel.inputs,
          outputs: beforeFinalLayer.output,
        });

        // Warm up the model
        this.tf.tidy(() => {
          const warmupInput = this.tf.zeros([1, 224, 224, 3]);
          const answer = this.featureExtractor.predict(warmupInput);
          answer.print();
        });

        console.log("MobileNet v2 feature extractor created successfully");
      } catch (error) {
        console.error(`Error setting up the feature extractor: ${error}`);
        throw error;
      }

      // define architecture of classification head
      this.fc_head = this.tf.sequential();
      this.fc_head.add(
        this.tf.layers.dense({
          inputShape: [this.featureExtractor.outputs[0].shape[1]],
          units: 64,
          activation: "relu",
        })
      );
      this.fc_head.add(
        this.tf.layers.dense({
          units: NUM_CLASSES,
          activation: "softmax",
        })
      );

      // Compile classification head
      const model_optimizer = this.tf.train.adam(0.01);
      this.fc_head.compile({
        optimizer: model_optimizer,
        loss:
          NUM_CLASSES === 2
            ? "binaryCrossentropy"
            : "categoricalCrossentropy",
        metrics: ["accuracy"],
      });

      console.log("Model created successfully");
      this.fc_head.summary();

      // preprocess image data and load as tensors
      const trainingData = await this.loadImages();
      const oneHotLabels = this.tf.oneHot(trainingData[1], NUM_CLASSES);

      const BATCH_SIZE = 3;
      const NUM_EPOCHS = 10;
      const start = performance.now();

      const results = await this.fc_head.fit(trainingData[0], oneHotLabels, {
        shuffle: true,
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS
      });

      const end = performance.now();
      const timeTakenInSeconds = ((end - start) / 1000).toFixed(2);
      console.log(`Training completed in ${timeTakenInSeconds} seconds`);

      // Clean up tensors
      trainingData[0].dispose();
      trainingData[1].dispose();
      oneHotLabels.dispose();

      this.trainingWithFeatureExtractor = false;

      console.log("Training results:", results.history);

      // save the model
      await this.fc_head.save("downloads://pretrained-head-v1");
    },
    train: async function (model, data) {
      const BATCH_SIZE = 3;
      const NUM_EPOCHS = 50;
      // const TRAIN_DATA_SIZE = 12;
      // const TEST_DATA_SIZE = 0;

      console.log("Fitting...");
      return model.fit(data[0], 
        data[1], {
        batchSize: BATCH_SIZE,
        epochs: NUM_EPOCHS
      });
    },
    runMain: async function () {                         // runMain function
      console.log("Button clicked...");
      this.checkFortf();

      const trainingData = await this.loadImages();
      /* below code should be moved to a new all-encompassing 
      'run' function
      */
      const convnet = this.getCNNModel();

      this.train(convnet, trainingData);
      console.log("Training complete...!");

      /*const savedModel = */await convnet.save("downloads://pretrained-model-v1");
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
</style>
