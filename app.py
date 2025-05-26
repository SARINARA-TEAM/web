from flask import Flask, render_template

app = Flask(__name__)

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/kluster')
def kluster():
    return render_template('kluster.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/detail_zat/<int:zat_id>')
def detail_zat(zat_id):
    details = {
        0: {
            "name": "Nikotin",
            "description": (
                "Nikotin adalah senyawa kimia yang sangat adiktif dan terdapat secara alami dalam tanaman tembakau. "
                "Zat ini bekerja langsung pada otak dan sistem saraf, memicu pelepasan dopamin yang memberikan sensasi senang sementara. "
                "Namun, nikotin juga dapat meningkatkan tekanan darah, denyut jantung, dan risiko gangguan jantung serta ketergantungan."
            )
        },
        1: {
            "name": "Tar",
            "description": (
                "Tar adalah zat residu lengket berwarna coklat gelap hingga hitam yang dihasilkan dari pembakaran rokok. "
                "Zat ini mengandung ribuan bahan kimia berbahaya, termasuk karsinogen (pemicu kanker) seperti benzena dan arsenik. "
                "Tar dapat menempel di paru-paru, merusak jaringan paru, menyebabkan iritasi saluran pernapasan, dan meningkatkan risiko kanker paru-paru serta penyakit pernapasan kronis."
            )
        },
        2: {
            "name": "Carbon Monoxide (CO)",
            "description": (
                "Carbon Monoxide (CO) adalah gas beracun tanpa warna dan bau yang dihasilkan dari pembakaran tembakau. "
                "Gas ini mengikat hemoglobin dalam darah lebih kuat daripada oksigen, sehingga mengurangi kemampuan darah mengangkut oksigen ke seluruh tubuh. "
                "Akibatnya, organ vital seperti jantung dan otak bisa kekurangan oksigen, meningkatkan risiko penyakit jantung, stroke, dan gangguan pernapasan."
            )
        },
        3: {
            "name": "Formaldehyde",
            "description": (
                "Formaldehyde adalah senyawa kimia beracun yang biasa digunakan sebagai bahan pengawet dalam industri medis dan kosmetik. "
                "Dalam asap rokok, formaldehyde terbentuk dari pembakaran zat organik. Zat ini bersifat iritan dan karsinogenik, dapat menyebabkan iritasi pada mata, hidung, tenggorokan, dan saluran pernapasan. "
                "Paparan jangka panjang formaldehyde dapat meningkatkan risiko kanker hidung, tenggorokan, dan saluran pernapasan bagian atas."
            )
        },
    }
    zat = details.get(zat_id, {"name": "Unknown", "description": "No details available."})
    return render_template('detail_zat.html', zat=zat)

if __name__ == '__main__':
    app.run(debug=True)
