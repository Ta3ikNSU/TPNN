import java.util.ArrayList;

public class Sigmoid {

    ArrayList<Float> input;
    ArrayList<Float> d_input;
    ArrayList<Float> output;
    ArrayList<Float> d_output;
    int units;

    Sigmoid(int units) {
        this.units = units;
        this.input = new ArrayList<>(this.units);
        this.d_input = new ArrayList<>(this.units);
        this.output = new ArrayList<>(this.units);
        this.d_output = new ArrayList<>(this.units);
    }

    void sigmoid() {
        for (int i = 0; i < units; i++) {
            output.set(i, (float) (1 / (1 + Math.exp(input.get(i)))));
        }
    }

    void sigmoid_back() {
        for (int i = 0; i < units; i++) {
            d_input.set(i, d_output.get(i) * (output.get(i) * (1 - output.get(i))));
        }
    }
}
