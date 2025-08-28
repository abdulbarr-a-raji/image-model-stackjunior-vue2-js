<template>
  <div class="training">
    <h1>{{ header }}</h1>
    <h4>CNN Model Demo</h4>
    <div id="micro-out-div">uses tf.js for ML</div>
    <br>
    <input type="file" id="imagedataUpload" multiple accept="image/*" />
    <button id="load-images" v-on:click="runMain">Load Image Data & Train Model</button>
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
      tf: null
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
    checkForTFJS: function () {
      if (this.tf) {
        console.log("tf.js is in this scope :D");
      } else {
        console.error("tf.js is NOT in this scope D':");
      }
    },
    preprocessImage: function (img) {
      return this.tf.tidy(() => {
        return this.tf.browser.fromPixels(img)
        .resizeBilinear([this.w, this.h])
        .toFloat()
        .div(this.tf.scalar(255));
      });
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
      this.checkForTFJS();

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
