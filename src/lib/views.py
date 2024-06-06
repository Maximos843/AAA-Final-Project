from aiohttp.web import Response
from aiohttp.web import View

from aiohttp_jinja2 import render_template

from src.lib.utils import open_image, preprocess_image, image_to_img_src, convert_description_to_tokens
from src.lib.config import Variables
from PIL import Image


class IndexView(View):
    template = 'index.html'

    async def get(self) -> Response:
        return render_template(self.template, self.request, {})

    async def post(self) -> Response:
        try:
            form = await self.request.post()
            image = open_image(form['image'].file)
            main_image = Image.fromarray(image.copy())
            image = preprocess_image(image)['image'].to(Variables.DEVICE)

            description = str(form['description'])
            vectorizer = self.request.app['vectorizer']
            description = convert_description_to_tokens(description, vectorizer['tokenizer'],
                                                        vectorizer['encode_mapping']).to(Variables.DEVICE)

            model = self.request.app['model']
            model.eval()
            #predictions = model(image.unsqueeze(0)).softmax(dim=1).detach().cpu()[0]
            predictions = model(image.unsqueeze(0), description.unsqueeze(0)).softmax(dim=1).detach().cpu()[0]
            result = []
            for i in range(len(predictions)):
                result.append({
                    'type': Variables.TARGETS[i],
                    'confidence': predictions[i],
                })
            main_image = image_to_img_src(main_image)
            result = sorted(result, key=lambda x: x['confidence'], reverse=True)
            ctx = {'main_image': main_image, 'predictions': result}
        except Exception as err:
            form = await self.request.post()
            description = str(form['description'])
            vectorizer = self.request.app['vectorizer']
            description = convert_description_to_tokens(description, vectorizer['tokenizer'],
                                                        vectorizer['encode_mapping']).to(Variables.DEVICE)
            ctx = {'error': str(err), 'error_type': type(err).__name__, 'desc': description, 'shape': description.shape}
        return render_template(self.template, self.request, ctx)
