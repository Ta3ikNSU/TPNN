import com.sun.jdi.IntegerType;

import java.util.*;

public class Linear {
    ArrayList<Float> input;
    ArrayList<Float> d_input;
    ArrayList<Float> output;
    ArrayList<Float> d_output;

    int element;

    Map<Integer, ArrayList<Float>> weight;
    int in_units, out_units;

    Linear(int in_units, int out_units) {
        this.in_units = in_units;
        this.out_units = out_units;
        this.input = new ArrayList<>(this.in_units);
        this.d_input = new ArrayList<>(this.in_units);
        this.output = new ArrayList<>(this.out_units);
        this.d_output = new ArrayList<>(this.out_units);
        this.weight = new HashMap<>(10);
        for (int i = 0; i < out_units; i++) {
            ArrayList<Float> list = new ArrayList<>();
            Random rand = new Random();
            for (int j = 0; j < in_units; j++) {
                list.add(rand.nextFloat(0.04f, 0.06f));
            }
            weight.put(i, list);
        }
    }


    void classification() {
        float sum = 0;
        for (int i = 0; i < out_units; i++) {
            float tempVal = 0;
            for (int j = 0; j < in_units; j++) {
                tempVal = tempVal + input.get(j) * weight.get(i).get(j);
            }
            output.set(i, tempVal);
            sum += tempVal;
        }

        for (int i = 0; i < out_units; i++) {
            output.set(i, output.get(i) / sum);
            if(Objects.equals(output.get(i), Collections.max(output))){
                element = i;
            }
        }
    }

}
