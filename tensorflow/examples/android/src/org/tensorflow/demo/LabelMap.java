package org.tensorflow.demo;

import android.content.res.AssetManager;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.Vector;

public class LabelMap {
  public Vector<String> loadFromAssets(final AssetManager assetManager,
                                       final String labelFilename) {
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    BufferedReader br = null;
    Vector<String> labels = new Vector<String>();
    try {
      br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
      String line;
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Problem reading label file!", e);
    }
    return labels;
  }

  public LabelMap () {};
}
