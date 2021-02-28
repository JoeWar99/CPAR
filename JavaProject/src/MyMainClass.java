import java.sql.SQLOutput;
import java.util.Arrays;
import java.util.Scanner;

public class MyMainClass {

    public void printResultAndTime(long start, int m_br, double[] phc){
        System.out.println("Result matrix: ");
        for(int i=0; i<1; i++)
        {	for(int j=0; j<Math.min(10,m_br); j++)
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


    public void onMult(int m_ar, int m_br){
        double[] pha = new double[m_ar * m_ar];
        double[] phb = new double[m_ar * m_ar];
        double[] phc = new double[m_ar * m_ar];
        int temp;

        Arrays.fill(pha, 1.0);

        // Get current time
        long start = System.currentTimeMillis();

        for(int i=0; i<m_br; i++)
            for(int j=0; j<m_br; j++)
                phb[i*m_br + j] = i+1;

        for(int i=0; i<m_ar; i++)
        {	for(int j=0; j<m_br; j++) {
                temp = 0;
                for(int k=0; k<m_ar; k++) {
                    temp += pha[i*m_ar+k] * phb[k*m_br+j];
                }
                phc[i*m_ar+j]=temp;
            }
        }

        printResultAndTime(start, m_br, phc);
    }

    public void onMultLine(int m_ar, int m_br){
        double[] pha = new double[m_ar * m_ar];
        double[] phb = new double[m_ar * m_ar];
        double[] phc = new double[m_ar * m_ar];
        int temp;

        Arrays.fill(pha, 1.0);

        // Get current time
        long start = System.currentTimeMillis();

        for(int i=0; i<m_br; i++)
            for(int j=0; j<m_br; j++)
                phb[i*m_br + j] = i+1;

        for(int i=0; i<m_ar; i++)
        {	for(int k=0; k<m_ar; k++) {
                for (int j = 0; j < m_br; j++) {
                    phc[i * m_ar + j] += pha[i * m_ar + k] * phb[k * m_br + j];
                }
            }
        }

        printResultAndTime(start, m_br, phc);
    }

    public static void main(String[] args) {
        MyMainClass mainClass = new MyMainClass();
        Scanner scan = new Scanner(System.in);
        int op, lin, col;

        op = 1;
        do {
            System.out.println("0. Quit Program");
            System.out.println("1. Multiplication");
            System.out.println("2. Line Multiplication");
            System.out.println("Selection?: ");
            op = scan.nextInt();

            if (op == 0)
                break;

            System.out.println("Dimensions: lins cols ? ");

            lin = scan.nextInt();
            col = scan.nextInt();

            switch (op){
                case 1:
                    mainClass.onMult(lin, col);
                    break;
                case 2:
                    mainClass.onMultLine(lin, col);
                    break;
            }
        }while (op != 0);

        scan.close();
    }

}
