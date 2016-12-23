import cv2
import decimal
import json
import ijson.backends.yajl2_cffi as ijson
from sklearn_theano.feature_extraction import OverfeatTransformer

tr = OverfeatTransformer(output_layers=[8])

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)

with open('../workspace/ds.json') as inh:
    with open('../workspace/ds_deep.json', 'w') as outh:
        ds = ijson.items(inh, 'item')
        outh.write('[')

        for i, item in enumerate(ds):
            print 'running', i+1
            if i > 0:
                outh.write(',')
            img = cv2.imread('set1/' + item['file'])
            img = cv2.resize(img, (231, 231))
            item['deep'] = tr.transform(img)[0].tolist()
            json.dump(item, outh, cls=DecimalEncoder)

        outh.write(']')