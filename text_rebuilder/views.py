from django.http import JsonResponse
from rest_framework.decorators import api_view
from .functions import extract_noun_phrases, generate_noun_permutations


@api_view(['GET'])
def paraphrase(request):
    try:
        tree_string = request.GET.get('tree', '')
        limit = int(request.GET.get('limit', '20'))
        nouns = extract_noun_phrases(tree_string)
        results = generate_noun_permutations(tree_string, nouns, limit)
        return JsonResponse({'paraphrased_trees': results}, json_dumps_params={'indent': 4, 'ensure_ascii': False})
    except Exception as e:
        return JsonResponse({'error': str(e)})


