import functools
import matplotlib.pyplot as plt
import pandas as pd


def average_calculate(filename):
    with open(filename, 'r') as f:
        content = f.read()

        data_set = content.splitlines()
        time = []
        Diff_xCordi = []
        Diff_yCordi = []
        Diff_angle = []
        for data in data_set:
            dt = data.strip().split(' ')
            time.append(abs(float(dt[0])))
            Diff_angle.append(abs(float(dt[3])))
            Diff_xCordi.append(abs(float(dt[1])))
            Diff_yCordi.append(abs(float(dt[2])))

        average_time = functools.reduce(lambda a, b: a + b, time) / len(time)
        average_diff_x = functools.reduce(lambda a, b: a + b, Diff_xCordi) / len(Diff_xCordi)
        average_diff_y = functools.reduce(lambda a, b: a + b, Diff_yCordi) / len(Diff_yCordi)
        average_diff_angle = functools.reduce(lambda a, b: a + b, Diff_angle) / len(Diff_angle)
        return average_time, average_diff_x, average_diff_y, average_diff_angle


if __name__ == '__main__':
    x_axis = [10, 50, 100, 250, 500, 1000]
    a_time = []
    a_x_cordi = []
    a_y_cordi = []
    a_angle = []
    filename = 'data10Particles.txt'
    a_t, a_x, a_y,a_a = average_calculate(filename)
    a_time.append(a_t)
    a_x_cordi.append(a_x)
    a_y_cordi.append(a_y)
    a_angle.append(a_a)
    print(a_t, a_x, a_y, a_a)
    filename = 'data50Particles.txt'
    a_t, a_x, a_y, a_a = average_calculate(filename)
    a_time.append(a_t)
    a_x_cordi.append(a_x)
    a_y_cordi.append(a_y)
    a_angle.append(a_a)
    print(a_t, a_x, a_y, a_a)
    filename = 'data100Particles.txt'
    a_t, a_x, a_y, a_a = average_calculate(filename)
    a_time.append(a_t)
    a_x_cordi.append(a_x)
    a_y_cordi.append(a_y)
    a_angle.append(a_a)
    print(a_t, a_x, a_y, a_a)
    filename = 'data250Particles.txt'
    a_t, a_x, a_y, a_a = average_calculate(filename)
    a_time.append(a_t)
    a_x_cordi.append(a_x)
    a_y_cordi.append(a_y)
    a_angle.append(a_a)
    print(a_t, a_x, a_y, a_a)
    filename = 'data500Particles.txt'
    a_t, a_x, a_y, a_a = average_calculate(filename)
    a_time.append(a_t)
    a_x_cordi.append(a_x)
    a_y_cordi.append(a_y)
    a_angle.append(a_a)
    print(a_t, a_x, a_y, a_a)
    filename = 'data1000Particles.txt'
    a_t, a_x, a_y, a_a = average_calculate(filename)
    a_time.append(a_t)
    a_x_cordi.append(a_x)
    a_y_cordi.append(a_y)
    a_angle.append(a_a)
    print(a_t, a_x, a_y, a_a)

    df = pd.DataFrame({'particle_number': x_axis, 'average_time': a_time})

    plt.plot('particle_number', 'average_time', data=df)
    plt.title('Das Verhältnis zwischen Partikelanzahl und Zeit-Performance')
    plt.xlabel('Partikelanzahl')
    plt.ylabel('Zeitperformance in Sekunden')
    #plt.show()
    plt.savefig('figure1')
    plt.clf()

    df2 = pd.DataFrame({'particle_number': x_axis, 'average_px': a_x_cordi})

    plt.plot('particle_number', 'average_px', data=df2)
    plt.title('Die Abhängigheit zwischen Partikelanzahl und Fehlertoloranz der Positionsschätzung')
    plt.xlabel('Partikelanzahl')
    plt.ylabel('Fehlertoloranz der X-Koordinate-Schätzung')
    # plt.show()
    plt.savefig('figure2')
    plt.clf()

    df3 = pd.DataFrame({'particle_number': x_axis, 'average_py': a_y_cordi})
    plt.plot('particle_number', 'average_py', data=df3)
    plt.title('Die Abhängigheit zwischen Partikelanzahl und Fehlertoloranz der Positionsschätzung')
    plt.xlabel('Partikelanzahl')
    plt.ylabel('Fehlertoloranz der Y-Koordinate-Schätzung')
    # plt.show()
    plt.savefig('figure3')
    plt.clf()

    df4 = pd.DataFrame({'particle_number': x_axis, 'average_angle': a_angle})
    plt.plot('particle_number', 'average_angle', data=df4)
    plt.title('Die Abhängigheit zwischen Partikelanzahl und Fehlertoloranz der Winkelsschätzung')
    plt.xlabel('Partikelanzahl')
    plt.ylabel('Fehlertoloranz der Winkelsschätzung')
    # plt.show()
    plt.savefig('figure4')
    plt.clf()




