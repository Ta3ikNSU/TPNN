import java.util.ArrayList;
import java.util.Random;

public class Convolutional {

    ArrayList<Float> input;
    ArrayList<Float> d_input;
    ArrayList<Float> output;
    ArrayList<Float> d_output;
    ArrayList<Float> weights;
    ArrayList<Float> d_weights;
    int in_units, out_units;
    int kernel_size;
    int stride = 1;
    int in_w, in_h, out_w, out_h;

    Convolutional(int kernel_size, int in_w, int in_h, int out_w, int out_h) {
        this.kernel_size = kernel_size;
        this.in_units = in_w * in_h;
        this.out_units = out_w * out_h;
        this.in_h = in_h;
        this.in_w = in_w;
        this.out_h = out_h;
        this.out_w = out_w;

        this.weights = new ArrayList<>(this.kernel_size * this.kernel_size);
        this.d_weights = new ArrayList<>(this.kernel_size * this.kernel_size);
        this.d_input = new ArrayList<>(this.in_units);
        this.input = new ArrayList<>(this.in_units);
        this.d_output = new ArrayList<>(this.out_units);
        this.output = new ArrayList<>(this.out_units);
        Random rand = new Random();
        weights.replaceAll(ignored -> rand.nextFloat(0.04f, 0.06f));
    }

    void conv() {
        int r = (in_h - kernel_size) / stride + 1;
        int c = (in_w - kernel_size) / stride + 1;
        for (int i = 0, x = 0; i < r; i++, x += stride) {
            for (int j = 0, y = 0; j < c; j++, y += stride) {
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        output.set(i * kernel_size + j, output.get(i * kernel_size + j) + input.get((m + x) * kernel_size + n + y) * weights.get(m * kernel_size + n));
                    }
                }
            }
        }
    }

    void conv_back() {
        ArrayList<Float> sigmas = new ArrayList<>(kernel_size * kernel_size);
        int r = in_h / stride + 1;
        int c = in_w / stride + 1;
        for (int i = 0, x = 0; i < r; i++, x += stride) {
            for (int j = 0, y = 0; j < c; j++, y += stride) {
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        float temp = sigmas.get(m * kernel_size + n);
                        sigmas.set(m * kernel_size + n, temp + d_output.get(i * kernel_size + j) * input.get((m + x) * kernel_size + n + y) * weights.get(m * kernel_size + n));
                    }
                }
            }
        }

        for (int m = 0; m < kernel_size; m++) {
            for (int n = 0; n < kernel_size; n++) {
                weights.set(m * kernel_size + n, weights.get(m * kernel_size + n) + sigmas.get(m * kernel_size + n));
            }
        }
    }

}
