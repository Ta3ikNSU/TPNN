import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Network {
    int in = 28;
    Convolutional convolutional1 = new Convolutional(5, in, in, in - 4, in - 4);
    ReLu reLu1 = new ReLu(convolutional1.out_units);

    AveragePooling averagePooling1 = new AveragePooling(3, convolutional1.out_h, convolutional1.out_w, convolutional1.out_h - 2, convolutional1.out_w - 2);

    Convolutional convolutional2 = new Convolutional(3, averagePooling1.out_h, averagePooling1.out_w, averagePooling1.out_h - 2, averagePooling1.out_w - 2);
    ReLu reLu2 = new ReLu(convolutional2.out_units);

    AveragePooling averagePooling2 = new AveragePooling(3, convolutional2.out_h, convolutional2.out_w, convolutional2.out_h - 2, convolutional2.out_w - 2);

    Convolutional convolutional3 = new Convolutional(3, averagePooling2.out_h, averagePooling2.out_w, averagePooling2.out_h - 2, averagePooling2.out_w - 2);
//    ReLu reLu3 = ReLu(convolutional3.out_units);

    Sigmoid sigmoid = new Sigmoid(convolutional3.out_units);

    Linear linear = new Linear(sigmoid.units, 10);

    HashMap<Integer, ArrayList<ArrayList<Float>>> trainset = new HashMap<>(10);
    ArrayList<ArrayList<Float>> errors;

    void init() {
        for (int i = 0; i < 10; i++) {
            trainset.put(i, getAllImagesBrightnessByNumber(i));
        }
    }

    void forward_backward(int number) {
        ArrayList<ArrayList<Float>> brightness = trainset.get(number);
        // Прогон для всех картинок одной цифры
        for (var bright : brightness) {
            convolutional1.input = (ArrayList<Float>) bright.clone();
            convolutional1.conv();

            reLu1.input = (ArrayList<Float>) convolutional1.output.clone();
            reLu1.relu();

            averagePooling1.input = (ArrayList<Float>) reLu1.output.clone();
            averagePooling1.pooling();

            convolutional2.input = (ArrayList<Float>) averagePooling1.output.clone();
            convolutional2.conv();

            reLu2.input = (ArrayList<Float>) convolutional2.output.clone();
            reLu2.relu();

            averagePooling2.input = (ArrayList<Float>) reLu2.output.clone();
            averagePooling2.pooling();

            convolutional3.input = (ArrayList<Float>) averagePooling2.output.clone();
            convolutional3.conv();

            sigmoid.input = (ArrayList<Float>) convolutional3.output.clone();
            sigmoid.sigmoid();

            linear.input = (ArrayList<Float>) sigmoid.output.clone();
            linear.classification();

            ArrayList<Float> output = (ArrayList<Float>) linear.output.clone();

            float error = 0;
            for (int i = 0; i < 10; i++) {
                if (number == i) {
                    error += (1 - output.get(number)) * (1 - output.get(number));
                } else {
                    error += output.get(i) * output.get(i);
                }
            }
            // ошибка выхода
            error += Math.sqrt(error);
            errors.get(number).add(error);

            linear.d_output = (ArrayList<Float>) linear.output.clone();
            linear.d_output.set(number, 1 - linear.d_output.get(number));

            sigmoid.d_output = (ArrayList<Float>) linear.d_input.clone();
            sigmoid.sigmoid_back();

            convolutional3.conv_back();
            convolutional3.d_output = (ArrayList<Float>) sigmoid.d_input.clone();

            reLu2.relu_back();
            reLu2.d_output = (ArrayList<Float>) convolutional3.input.clone();

            convolutional2.conv_back();
            convolutional2.d_output = (ArrayList<Float>) reLu2.d_input.clone();

            reLu1.relu_back();
            reLu2.d_output = (ArrayList<Float>) convolutional2.input.clone();

            convolutional1.conv_back();
            convolutional1.d_output = (ArrayList<Float>) reLu1.d_input.clone();
        }
    }

    public void train(int epochs) {
        errors = new ArrayList<>(10);
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Прогон для всех цифр
            for (int number = 0; number < 10; number++) {
                forward_backward(number);
            }
        }
        errors.forEach(System.out::println);
    }

    ArrayList<ArrayList<Float>> getAllImagesBrightnessByNumber(int number) {
        try {
            List<File> filesInFolder = Files.walk(Paths.get("src/MNIST - JPG - training/" + number)).filter(Files::isRegularFile).map(Path::toFile).toList();
            ArrayList<ArrayList<Float>> brightness = new ArrayList<>(filesInFolder.size());
            filesInFolder.forEach((file -> {
                ArrayList<Float> bright = new ArrayList<>(28 * 28);
                try {
                    BufferedImage img = ImageIO.read(file);
                    for (int i = 0; i < img.getWidth(); i++) {
                        for (int j = 0; j < img.getHeight(); j++) {
                            int color = img.getRGB(i, j);
                            int red = (color >>> 16) & 0xFF;
                            int green = (color >>> 8) & 0xFF;
                            int blue = (color) & 0xFF;
                            float luminance = (red * 0.2126f + green * 0.7152f + blue * 0.0722f) / 255;
                            bright.set(i * img.getHeight() + j, luminance);
                        }
                    }
                    brightness.add(bright);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }));
            return brightness;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new ArrayList<>();
    }
}
