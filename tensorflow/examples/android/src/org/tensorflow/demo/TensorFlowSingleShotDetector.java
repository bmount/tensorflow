/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.env.SplitTimer;

/**
 * An object detection example using the method described in "SSD: Single Shot MultiBox Detector"
 * (https://arxiv.org/abs/1512.02325)
 */
public class TensorFlowSingleShotDetector implements Classifier {
  private static final Logger LOGGER = new Logger();

  private Vector<String> labels = new Vector<String>();

  // Config values.
  private String inputName;
  private int inputSize;

  // Pre-allocated buffers.
  private int[] intPixels;
  private byte[] bytePixels;
  private String[] outputNames;

  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  /** Initializes a native TensorFlow session for classifying images. */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelsFilename,
      final int inputSize,
      final String inputName,
      final String outputNamesConcatenated) {
    TensorFlowSingleShotDetector d = new TensorFlowSingleShotDetector();
    d.inputName = inputName;
    d.inputSize = inputSize;

    // Pre-allocate buffers.
    d.outputNames = outputNamesConcatenated.split(",");
    d.intPixels = new int[inputSize * inputSize];
    d.bytePixels = new byte[inputSize * inputSize * 3];

    d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);
    d.labels = (new LabelMap()).loadFromAssets(assetManager, labelsFilename);

    return d;
  }

  private TensorFlowSingleShotDetector() {}

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    final SplitTimer timer = new SplitTimer("recognizeImage");

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");

    bitmap.getPixels(intPixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    // (TODO) this shouldn't be necessary
    for (int i = 0; i < intPixels.length; ++i) {
      bytePixels[i * 3 + 0] = (byte)(intPixels[i] & 0xFF);
      bytePixels[i * 3 + 1] = (byte)((intPixels[i] >> 8) & 0xFF);
      bytePixels[i * 3 + 2] = (byte)((intPixels[i] >> 16) & 0xFF);
    }

    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    inferenceInterface.feed(inputName, bytePixels, 1, inputSize, inputSize, 3);
    Trace.endSection(); // feed

    timer.endSplit("ready for inference");

    // Run the inference call.
    Trace.beginSection("run");
    inferenceInterface.run(outputNames, logStats);
    Trace.endSection();

    timer.endSplit("ran inference");

    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");

    final float[] numDetectionsArray = new float[1];
    inferenceInterface.fetch(outputNames[3], numDetectionsArray);
    final int numDetections = Math.round(numDetectionsArray[0]);
    final float[] boxes = new float[numDetections * 4];
    final float[] classes = new float[numDetections];
    final float[] scores = new float[numDetections];
    inferenceInterface.fetch(outputNames[0], boxes);
    inferenceInterface.fetch(outputNames[1], scores);
    inferenceInterface.fetch(outputNames[2], classes);

    Trace.endSection(); // fetch

    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
    int w = bitmap.getWidth();
    int h = bitmap.getHeight();
    for (int i = 0; i < numDetections; i++) {
      if (scores[i] > 0.07) {
        int boxIdx = i * 4;
        float x0 = boxes[boxIdx + 1]; // tf convention is (y, x)
        float y0 = boxes[boxIdx + 0];
        float x1 = boxes[boxIdx + 3];
        float y1 = boxes[boxIdx + 2];
        final RectF rect = new RectF(
                  Math.max(0, x0 * w),
                  Math.max(0, y0 * h),
                  Math.min(bitmap.getWidth() - 1, x1 * w),
                  Math.min(bitmap.getHeight() - 1, y1 * h));
        int cls = (int)classes[i];
        recognitions.add(new Recognition("" + i, labels.get(cls), scores[i], rect));
      }
    }

    Trace.endSection(); // "recognizeImage"

    timer.endSplit("processed results");

    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
