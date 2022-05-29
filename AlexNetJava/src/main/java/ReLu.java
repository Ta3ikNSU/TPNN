import java.util.ArrayList;

public class ReLu {
    ArrayList<Float> input;
    ArrayList<Float> d_input;
    ArrayList<Float> output;
    ArrayList<Float> d_output;
    int units;

    ReLu(int units) {
        this.units = units;
        input = new ArrayList<>(this.units);
        d_input = new ArrayList<>(this.units);
        output = new ArrayList<>(this.units);
        d_output = new ArrayList<>(this.units);
    }


    void relu() {
        for (int i = 0; i < units; i++) {
            if (input.get(i) > 64) {
                output.set(i, input.get(i));
            } else {
                output.set(i, 0f);
            }
        }
    }

    void backward() {
        for (int i = 0; i < units; i++) {
            d_input.set(i, d_output.get(i) * input.get(i));
        }
    }
}