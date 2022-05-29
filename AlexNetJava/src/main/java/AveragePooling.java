import java.util.ArrayList;

public class AveragePooling {

    ArrayList<Float> input;
    ArrayList<Float> d_input;
    ArrayList<Float> output;
    ArrayList<Float> d_output;
    int in_units, out_units;
    int kernel_size;
    int stride = 1;
    int in_w, in_h, out_w, out_h;

    AveragePooling(int kernel_size, int in_w, int in_h, int out_w, int out_h) {
        this.in_units = in_w * in_h;
        this.out_units = out_w * out_h;
        this.kernel_size = kernel_size;
        this.input = new ArrayList<>(this.in_units);
        this.d_input = new ArrayList<>(this.in_units);
        this.output = new ArrayList<>(this.out_units);
        this.d_output = new ArrayList<>(this.out_units);
        this.in_w = in_w;
        this.in_h = in_h;
        this.out_w = out_w;
        this.out_h = out_h;
    }

    void pooling() {
        int r = (in_h - kernel_size) / stride + 1;
        int c = (in_w - kernel_size) / stride + 1;
        for (int i = 0, x = 0; i < r; i++, x += stride) {
            for (int j = 0, y = 0; j < c; j++, y += stride) {
                float sum = 0;
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        sum += input.get((m + x) * kernel_size + n + y);
                    }
                }
                output.set(i * kernel_size + j, sum / (kernel_size * kernel_size));
            }
        }
    }
}
