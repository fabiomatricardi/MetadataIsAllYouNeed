import gradio as gr
from keybert import KeyBERT
#from ctransformers import AutoModelForCausalLM, AutoConfig, Config #import for GGML models
import datetime

doc = """
La filosofia della scienza è lo studio critico e riflessivo delle fondamenta, metodi e implicazioni di ogni branca della conoscenza scientifica. Si occupa di questioni come la natura della verità scientifica, le limitazioni del metodo scientifico, l'origine e il significato dell'interesse per la scoperta e il progresso scientifico, i rapporti tra scienza e tecnologia, la relazione tra scienza e filosofia stessa.
La filosofia della scienza ha origine nelle prime riflessioni dei filosofi antichi sulla natura dell'esperimento scientifico e sull'uso delle teorie scientifiche per spiegare il mondo. Tra i primi pensatori che si occuparono di queste questioni ci furono Aristotele, Platone e Democrito.
Nell'età moderna, la filosofia della scienza ha preso forma con le riflessioni dei grandi pensatori come Descartes, Bacon e Galilei sul metodo scientifico e sulla natura del progresso della conoscenza. In particolare, il metodo scientifico di Galilei è stato uno dei primi tentativi sistematici di definire le regole per la scoperta scientifica.
Nel XX secolo, la filosofia della scienza ha avuto un ulteriore sviluppo con l'affermazione del positivismo logico e il dibattito tra realisti e antirealisti sul significato delle teorie scientifiche. I principali pensatori in questo campo sono stati Karl Popper, Thomas Kuhn e Imre Lakatos.
In conclusione, la filosofia della scienza è un'area di studio critica che si occupa dei fondamenti epistemologici delle diverse branche della conoscenza scientifica. Ha avuto origine nelle riflessioni antiche sul metodo scientifico e ha continuato a svilupparsi nel XX secolo attraverso i dibattiti tra realisti, positivisti logici e antirealisti sulla natura del progresso scientifico.
"""


#MODEL SETTINGS also for DISPLAY
modelfile = "multi-qa-MiniLM-L6-cos-v1"
repetitionpenalty = 1.15
contextlength=4096
logfile = 'KeyBERT-promptsPlayground.txt'
print("loading model...")
stt = datetime.datetime.now()
from keybert import KeyBERT
kw_model = KeyBERT(model='multi-qa-MiniLM-L6-cos-v1')
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

"""
gr.themes.Base()
gr.themes.Default()
gr.themes.Glass()
gr.themes.Monochrome()
gr.themes.Soft()
"""
def combine(text, ngram,dvsity):
    import datetime
    import random
    a = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, ngram), stop_words='english',
                              use_mmr=True, diversity=dvsity, highlight=True)     
    output = ''' '''    
    colors = ['primary', 'secondary', 'success', 'danger','warning','info','light','dark']
    for kw in a:
        s = random.randint(0,6)
        output = output + f'''<span class="badge text-bg-{colors[s]}">{str(kw[0])}</span>
        
        '''
    STYLE = '''<head>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">
    </head>

'''
    OUTPUT_OK = STYLE  + '''<div class="container">
    '''  + output  + "</div><br>"
    
    #print(OUTPUT_OK)

    return OUTPUT_OK

    #return generation, delta


# MAIN GRADIO INTERFACE
with gr.Blocks(theme=gr.themes.Soft()) as demo:   #theme=gr.themes.Glass()
    #TITLE SECTION
    with gr.Row(variant='compact'):
        with gr.Column(scale=1):
            gr.HTML("<center><img src='https://github.com/fabiomatricardi/GradioStudies/raw/main/windows/bert.png' width=90></center>")
        with gr.Column(scale=3):
            gr.HTML("<center>"
            + "<p>Prompt Engineering Playground - Test your favourite LLM for advanced inferences</p>"
            + "<h1> KeyBERT keyword extraction</h1></center>")                    

    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
            f"""
            ### Tunning Parameters""")
            ngramrange = gr.Slider(label="nGram Range (1 to...)",minimum=1, maximum=4, step=1, value=2)
            diversity = gr.Slider(label="Text diversity", minimum=0.0, maximum=1.0, step=0.02, value=0.2)
            gr.Markdown(
            """
            Change the NNGram range, the num of words in the keyword
            Change the diversity, lower number is little diversity
            And then click the Button below
            """)
            btn = gr.Button(value="Generate Keywords", size='lg', icon='./bert.png')           
        with gr.Column(scale=4):
            text = gr.Textbox(label="Text to Analyze", lines=8)
            with gr.Group():
                gr.Markdown("""
                            >  Keywords""")
                labelled = gr.HTML()
            btn.click(combine, inputs=[text,ngramrange,diversity], outputs=[labelled])

if __name__ == "__main__":
    demo.launch(inbrowser=True)




