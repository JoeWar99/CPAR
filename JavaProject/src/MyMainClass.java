import java.sql.SQLOutput;
import java.util.Arrays;
import java.util.Scanner;

public class MyMainClass {

    public void printResultAndTime(long start, int size, double[] phc){
        System.out.println("Result matrix: ");
        for(int i=0; i<1; i++)
        {	for(int j=0; j<Math.min(10,size); j++)
            System.out.print(phc[j] + " ");
        }
        System.out.println("");

        // Get elapsed time in milliseconds
        long elapsedTimeMillis = System.currentTimeMillis()-start;
        // Get elapsed time in seconds
        float elapsedTimeSec = elapsedTimeMillis/1000F;
        // Get elapsed time in minutes
        float elapsedTimeMin = elapsedTimeMillis/(60*1000F);

        System.out.println("Time passed in milliseconds: " + elapsedTimeMillis);
        System.out.println("Time passed in seconds: " + elapsedTimeSec);
        System.out.println("Time passed in minutes: " + elapsedTimeMin);
    }


    public void onMult(int size){
        double[] pha = new double[size * size];
        double[] phb = new double[size * size];
        double[] phc = new double[size * size];
        int temp;
        
        Arrays.fill(pha, 1.0);
        Arrays.fill(phc, 0.0);

        // Get current time
        long start = System.currentTimeMillis();

        for(int i=0; i<size; i++)
            for(int j=0; j<size; j++)
                phb[i*size + j] = i+1;

        for(int i=0; i<size; i++)
        {	for(int j=0; j<size; j++) {
                temp = 0;
                for(int k=0; k<size; k++) {
                    temp += pha[i*size+k] * phb[k*size+j];
                }
                phc[i*size+j]=temp;
            }
        }

        printResultAndTime(start, size, phc);
    }

    public void onMultLine(int size){
        double[] pha = new double[size * size];
        double[] phb = new double[size * size];
        double[] phc = new double[size * size];
        int temp;
        
        Arrays.fill(pha, 1.0);
        Arrays.fill(phc, 0.0);

        // Get current time
        long start = System.currentTimeMillis();

        for(int i=0; i<size; i++)
            for(int j=0; j<size; j++)
                phb[i*size + j] = i+1;

        for(int i=0; i<size; i++)
        {	for(int k=0; k<size; k++) {
                for (int j = 0; j < size; j++) {
                    phc[i * size + j] += pha[i * size + k] * phb[k * size + j];
                }
            }
        }

        printResultAndTime(start, size, phc);
    }

    public static void main(String[] args) {
        MyMainClass mainClass = new MyMainClass();
        Scanner scan = new Scanner(System.in);
        int op, size;

        op = 1;
        do {
            System.out.println("0. Quit Program");
            System.out.println("1. Multiplication");
            System.out.println("2. Line Multiplication");
            System.out.println("Selection?: ");
            op = scan.nextInt();

            if (op == 0)
                break;

            System.out.println("Matrix Size ?");

            size = scan.nextInt();

            switch (op){
                case 1:
                    mainClass.onMult(size);
                    break;
                case 2:
                    mainClass.onMultLine(size);
                    break;
            }
        }while (op != 0);

        scan.close();
    }

}
